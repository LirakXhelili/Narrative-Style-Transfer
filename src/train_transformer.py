
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report


MODEL_NAME = "distilbert-base-uncased"
RANDOM_SEED = 42
MAX_LENGTH = 256


class StyleTransferDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
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
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    df = pd.read_pickle("data/processed/narrative_cues.pkl")

    texts = df["text"].tolist()
    labels = df["has_transfer"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = StyleTransferDataset(X_train, y_train, tokenizer)
    test_ds = StyleTransferDataset(X_test, y_test, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    Path("models").mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir="models/transformer_out",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=RANDOM_SEED,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    y_pred = logits.argmax(axis=-1)

    print("=== Transformer: binary style-transfer detection ===")
    print(classification_report(y_test, y_pred, digits=3))

   
    model.save_pretrained("models/transformer_style_transfer")
    tokenizer.save_pretrained("models/transformer_style_transfer")
    print("Saved transformer model to models/transformer_style_transfer")


if __name__ == "__main__":
    main()
