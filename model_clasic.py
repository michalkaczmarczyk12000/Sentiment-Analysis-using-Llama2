import os
import warnings
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline
)
from trl import SFTTrainer, setup_chat_format
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cfg.paths.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device
    )

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
    training_args = TrainingArguments(
        output_dir=cfg.paths.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        ddp_find_unused_parameters=False, 
        warmup_steps=cfg.training.warmup_steps,
        evaluation_strategy=cfg.training.evaluation_strategy,
        logging_dir=cfg.paths.logs_dir,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        report_to=cfg.training.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
    evaluate(y_true, y_pred, "metricsy.txt")


    trainer.save_model(cfg.paths.trained_model_dir)
    tokenizer.save_pretrained(cfg.paths.trained_model_dir)

if __name__ == "__main__":
    main()
