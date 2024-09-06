import os
import warnings
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from tqdm import tqdm
import bitsandbytes as bnb
import torch
from peft import LoraConfig, PeftConfig, get_peft_model
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline
)
from trl import SFTTrainer, setup_chat_format
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from accelerate import PartialState

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
    os.environ["TRANSFORMERS_CACHE"] = cfg.paths.cache_dir
    os.environ["HF_HOME"] = cfg.paths.cache_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(
        cfg.paths.data_file, names=["sentiment", "text"], encoding="utf-8", encoding_errors="replace"
    )

    X_train = list()
    X_test = list()
    for sentiment in ["positive", "neutral", "negative"]:
        train, test = train_test_split(
            df[df.sentiment == sentiment], train_size=300, test_size=300, random_state=42
        )
        X_train.append(train)
        X_test.append(test)

    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    X_test = pd.concat(X_test)

    eval_idx = [
        idx for idx in df.index if idx not in list(X_train.index) + list(X_test.index)
    ]
    X_eval = df[df.index.isin(eval_idx)]
    X_eval = X_eval.groupby("sentiment", group_keys=False).apply(
        lambda x: x.sample(n=50, random_state=10, replace=True)
    )
    X_train = X_train.reset_index(drop=True)

    def generate_prompt(data_point):
        return f"""
                Analyze the sentiment of the news headline enclosed in square brackets,
                determine if it is positive, neutral, or negative, and return the answer as
                the corresponding sentiment label "positive" or "neutral" or "negative".

                [{data_point["text"]}] = {data_point["sentiment"]}
                """.strip()

    def generate_test_prompt(data_point):
        return f"""
                Analyze the sentiment of the news headline enclosed in square brackets,
                determine if it is positive, neutral, or negative, and return the answer as
                the corresponding sentiment label "positive" or "neutral" or "negative".

                [{data_point["text"]}] = """.strip()

    X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), columns=["text"])
    X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), columns=["text"])

    y_true = X_test.sentiment
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

    train_dataset = Dataset.from_pandas(X_train)
    val_dataset = Dataset.from_pandas(X_eval)

    def evaluate(y_true, y_pred, output_file):
        labels = ["positive", "neutral", "negative"]
        mapping = {"positive": 2, "neutral": 1, "none": 1, "negative": 0}

        def map_func(x):
            return mapping.get(x, 1)

        y_true = np.vectorize(map_func)(y_true)
        y_pred = np.vectorize(map_func)(y_pred)

        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        unique_labels = set(y_true)
        label_accuracies = {}

        for label in unique_labels:
            label_indices = [i for i in range(len(y_true)) if y_true[i] == label]
            label_y_true = [y_true[i] for i in label_indices]
            label_y_pred = [y_pred[i] for i in label_indices]
            label_accuracy = accuracy_score(label_y_true, label_y_pred)
            label_accuracies[label] = label_accuracy

        class_report = classification_report(y_true=y_true, y_pred=y_pred)
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])

        with open(output_file, "w") as f:
            f.write(f"Accuracy: {accuracy:.3f}\n")
            for label, label_accuracy in label_accuracies.items():
                f.write(f"Accuracy for label {label}: {label_accuracy:.3f}\n")
            f.write("\nClassification Report:\n")
            f.write(class_report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(conf_matrix))

    model_name = cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer, cache_dir=cfg.paths.cache_dir)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.bnb.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.bnb.bnb_4bit_use_double_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=getattr(torch, cfg.bnb.bnb_4bit_compute_dtype),
        quantization_config=bnb_config,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, tokenizer = setup_chat_format(model, tokenizer)

    def predict(test, model, tokenizer):
        y_pred = []
        for i in tqdm(range(len(test))):
            prompt = test.iloc[i]["text"]
            pipe = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1,
                temperature=0.0,
            )
            result = pipe(prompt)
            answer = result[0]["generated_text"].split("=")[-1]
            if "positive" in answer:
                y_pred.append("positive")
            elif "negative" in answer:
                y_pred.append("negative")
            elif "neutral" in answer:
                y_pred.append("neutral")
            else:
                y_pred.append("none")
        return y_pred

    data_collator = DataCollatorWithPadding(tokenizer)
    print(train_dataset[0])

    peft_config = LoraConfig(
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        r=cfg.lora.r,
        bias=cfg.lora.bias,
        target_modules=cfg.lora.target_modules,
        task_type=cfg.lora.task_type,
    )
    model = get_peft_model(model, peft_config)

    training_arguments = TrainingArguments(
        output_dir=cfg.paths.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_steps=cfg.training.save_steps,
        logging_steps=cfg.training.logging_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        max_grad_norm=cfg.training.max_grad_norm,
        max_steps=cfg.training.max_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        group_by_length=cfg.training.group_by_length,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        report_to=cfg.training.report_to,
        evaluation_strategy=cfg.training.evaluation_strategy,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=1024,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        }
    )

    trainer.train()
    trainer.evaluate()
    y_pred = predict(X_test, model, tokenizer)
    evaluate(y_true, y_pred, "metrics.txt")

    trainer.save_model(cfg.paths.output_dir)
    tokenizer.save_pretrained(cfg.paths.output_dir)

if __name__ == "__main__":
    main()
