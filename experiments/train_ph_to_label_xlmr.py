#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e-SNLI Indonesia — Premis + Hipotesis (kalimat1 + kalimat2) -> Label dengan XLM-RoBERTa
(baseline NLI standar, memakai Hugging Face Trainer)

Input  : pasangan kalimat1 (premis) dan kalimat2 (hipotesis), tanpa penjelasan.
         Ditokenisasi sebagai sentence pair:  <s> premis </s></s> hipotesis </s>
Output : label emas (keterlibatan / netral / kontradiksi)

Format data (jsonl), kolom yang dipakai:
    kalimat1, kalimat2, label emas

Dependensi:
    pip install "transformers>=4.40" datasets accelerate scikit-learn pandas torch

Contoh:
    python train_ph_to_label_xlmr.py \
        --train_file esnli_indo/train_data.jsonl \
        --dev_file   esnli_indo/validation_data.jsonl \
        --test_file  esnli_indo/test_data.jsonl
"""

import argparse
import inspect
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

LABEL_NAMES = ["keterlibatan", "netral", "kontradiksi"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# --------------------------------------------------------------------------- #
# 1. Arsitektur model ([CLS] -> Linear-ReLU-Dropout-Linear)
#    Kompatibel dengan Trainer: menerima `labels` dan mengembalikan loss.
# --------------------------------------------------------------------------- #
class XLMRobertaPHClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_labels=3, dropout=0.1):
        super().__init__()
        self.xlm_roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.xlm_roberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        r_ph = out.last_hidden_state[:, 0, :]            # vektor [CLS] dari premis + hipotesis
        logits = self.classifier(r_ph)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


# --------------------------------------------------------------------------- #
# 2. Utilitas data
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    df = pd.read_json(path, lines=True)
    print(f"✅ {path}: {len(df)} baris, kolom = {list(df.columns)}")
    return df


def clean_split(df, label_col, premise_col, hypothesis_col, name):
    """Map label ke int dan BUANG baris berlabel '-' / premis atau hipotesis kosong."""
    for col in (premise_col, hypothesis_col, label_col):
        if col not in df.columns:
            raise KeyError(f"[{name}] kolom '{col}' tidak ada. Kolom: {list(df.columns)}")
    out = pd.DataFrame({
        "premise": df[premise_col],
        "hypothesis": df[hypothesis_col],
        "label_str": df[label_col].astype(str).str.strip().str.lower(),
    })
    n0 = len(out)
    out = out[out["label_str"].isin(LABEL2ID)]
    out = out.dropna(subset=["premise", "hypothesis"])
    for c in ("premise", "hypothesis"):
        out[c] = out[c].astype(str).str.strip()
        out = out[out[c] != ""]
    out["labels"] = out["label_str"].map(LABEL2ID).astype(int)
    print(f"[{name}] {n0 - len(out)} baris dibuang (label '-'/tidak valid atau teks kosong), "
          f"sisa {len(out)}")
    return out.reset_index(drop=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


# --------------------------------------------------------------------------- #
# 3. Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default="esnli_indo/train_data.jsonl")
    p.add_argument("--dev_file", default="esnli_indo/validation_data.jsonl")
    p.add_argument("--test_file", default="esnli_indo/test_data.jsonl")
    p.add_argument("--label_col", default="label emas")
    p.add_argument("--premise_col", default="kalimat1")
    p.add_argument("--hypothesis_col", default="kalimat2")
    p.add_argument("--model_name", default="xlm-roberta-base")
    p.add_argument("--output_dir", default="outputs/ph_to_label_xlmr")
    p.add_argument("--max_length", type=int, default=128,
                   help="Panjang maksimal gabungan premis + hipotesis (token).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=2, help="Early stopping per epoch; 0 = nonaktif")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    cols = (args.label_col, args.premise_col, args.hypothesis_col)

    # --- Data ---
    raw = {
        "train": load_jsonl(args.train_file),
        "validation": load_jsonl(args.dev_file),
        "test": load_jsonl(args.test_file),
    }
    train_df = clean_split(raw["train"], *cols, "train")
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)),
                                   random_state=args.seed).reset_index(drop=True)
    val_df = clean_split(raw["validation"], *cols, "validation")
    print("Distribusi label train:\n", train_df["label_str"].value_counts())

    # --- Tokenisasi pasangan kalimat (padding dinamis) ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        # truncation=True -> 'longest_first': memotong kalimat yang lebih panjang lebih dulu
        return tokenizer(batch["premise"], batch["hypothesis"],
                         truncation=True, max_length=args.max_length)

    def to_ds(df):
        return Dataset.from_pandas(df[["premise", "hypothesis", "labels"]]).map(
            tokenize, batched=True, remove_columns=["premise", "hypothesis"])

    dataset = DatasetDict({"train": to_ds(train_df), "validation": to_ds(val_df)})
    print(dataset)
    print("Contoh input ter-tokenisasi:",
          tokenizer.decode(dataset["train"][0]["input_ids"]))

    # --- Model & Trainer ---
    model = XLMRobertaPHClassifier(args.model_name, num_labels=len(LABEL_NAMES))

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # transformers 4.x: warmup_ratio; transformers 5.x: warmup_steps < 1 dianggap rasio.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    warmup_kw = ({"warmup_ratio": args.warmup_ratio} if "warmup_ratio" in ta_params
                 else {"warmup_steps": args.warmup_ratio})

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        **warmup_kw,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=use_fp16,
        bf16=use_bf16,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(args.patience)] if args.patience > 0 else [],
    )

    print(f"Memulai pelatihan di device: {training_args.device}")
    trainer.train()

    best_path = os.path.join(args.output_dir, "best_ph_to_label_xlmr_model.pt")
    torch.save(trainer.model.state_dict(), best_path)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Bobot model terbaik disimpan di {best_path}")

    # --- Evaluasi validasi & test ---
    results = {"config": vars(args)}
    labels_idx = list(range(len(LABEL_NAMES)))
    for split in ["validation", "test"]:
        df = clean_split(raw[split], *cols, split)
        out = trainer.predict(to_ds(df))
        y_pred = out.predictions.argmax(-1)
        y_true = df["labels"].values

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

        print("\n" + "=" * 55)
        print(f"   Laporan Kinerja — {split.upper()} (kalimat1 + kalimat2)")
        print("=" * 55)
        print(f"Akurasi         : {acc:.4f}")
        print(f"F1-Score (Macro): {f1:.4f}")
        print("-" * 55)
        print(classification_report(y_true, y_pred, labels=labels_idx,
                                    target_names=LABEL_NAMES, digits=4, zero_division=0))
        print("Confusion matrix (baris=gold, kolom=prediksi):", LABEL_NAMES)
        print(cm)

        results[split] = {
            "accuracy": float(acc), "macro_f1": float(f1),
            "per_class": classification_report(
                y_true, y_pred, labels=labels_idx, target_names=LABEL_NAMES,
                output_dict=True, zero_division=0),
            "confusion_matrix": cm.tolist(),
        }

        pred_df = df[["premise", "hypothesis", "label_str"]].copy()
        pred_df["prediksi"] = [ID2LABEL[i] for i in y_pred]
        probs = torch.softmax(torch.tensor(out.predictions), dim=-1).numpy()
        for i, lab in ID2LABEL.items():
            pred_df[f"prob_{lab}"] = probs[:, i]
        pred_df.to_csv(os.path.join(args.output_dir, f"pred_{split}.csv"), index=False)

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSelesai. Hasil di {args.output_dir}/results.json")


if __name__ == "__main__":
    main()