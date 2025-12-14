from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from .config import LABELS, resolve_processed_pkl
from .dataset import get_multilabel_targets


MODEL_NAME = "distilbert-base-uncased"
RANDOM_SEED = 42
MAX_LENGTH = 256


class StyleTransferDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
       
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))         
    preds = (probs >= 0.5).astype(int)       

    micro = f1_score(labels, preds, average="micro", zero_division=0)
    macro = f1_score(labels, preds, average="macro", zero_division=0)

    per_label = {}
    for i, name in enumerate(LABELS):
        per_label[f"f1_{name}"] = f1_score(labels[:, i], preds[:, i], zero_division=0)

    return {"micro_f1": micro, "macro_f1": macro, **per_label}


def main():
    df = pd.read_pickle(resolve_processed_pkl())

    texts = df["text"].tolist()
    y = get_multilabel_targets(df)  # (N, 4)

 
    strat = df["has_transfer"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=strat,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = StyleTransferDataset(X_train, y_train, tokenizer)
    test_ds = StyleTransferDataset(X_test, y_test, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )

    Path("models").mkdir(exist_ok=True)

  
    training_args = TrainingArguments(
        output_dir="models/transformer_out",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=RANDOM_SEED,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()


    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)

    print("=== Transformer: MULTI-LABEL cue detection ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
    )
    print("Micro-F1:", f1_score(y_test, y_pred, average="micro", zero_division=0))
    print("Macro-F1:", f1_score(y_test, y_pred, average="macro", zero_division=0))

    out_dir = Path("models/transformer_cues_multilabel")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved transformer model to {out_dir.as_posix()}")


if __name__ == "__main__":
    main()