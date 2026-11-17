#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e-SNLI Indonesia — Premis + Hipotesis + Penjelasan -> Label dengan XLM-RoBERTa
(memakai Hugging Face Trainer)

Input  : kalimat1 (premis), kalimat2 (hipotesis), dan penjelasan, digabung jadi
         satu urutan dengan token pemisah XLM-R:
             <s> premis </s></s> hipotesis </s></s> penjelasan </s>
Output : label emas (keterlibatan / netral / kontradiksi)

Format data (jsonl), kolom yang dipakai:
    kalimat1, kalimat2, penjelasan, label emas

Dependensi:
    pip install "transformers>=4.40" datasets accelerate scikit-learn pandas torch

Contoh:
    python train_phe_to_label_xlmr.py \
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
SEGMENTS = ["premise", "hypothesis", "explanation"]


# --------------------------------------------------------------------------- #
# 1. Arsitektur model ([CLS] -> Linear-ReLU-Dropout-Linear)
#    Kompatibel dengan Trainer: menerima `labels` dan mengembalikan loss.
# --------------------------------------------------------------------------- #
class XLMRobertaPHEClassifier(nn.Module):
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
        r_phe = out.last_hidden_state[:, 0, :]   # vektor [CLS] dari premis + hipotesis + penjelasan
        logits = self.classifier(r_phe)
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


def clean_split(df, label_col, premise_col, hypothesis_col, expl_col, name):
    """Map label ke int dan BUANG baris berlabel '-' / premis, hipotesis, atau penjelasan kosong."""
    for col in (premise_col, hypothesis_col, expl_col, label_col):
        if col not in df.columns:
            raise KeyError(f"[{name}] kolom '{col}' tidak ada. Kolom: {list(df.columns)}")
    out = pd.DataFrame({
        "premise": df[premise_col],
        "hypothesis": df[hypothesis_col],
        "explanation": df[expl_col],
        "label_str": df[label_col].astype(str).str.strip().str.lower(),
    })
    n0 = len(out)
    out = out[out["label_str"].isin(LABEL2ID)]
    out = out.dropna(subset=SEGMENTS)
    for c in SEGMENTS:
        out[c] = out[c].astype(str).str.strip()
        out = out[out[c] != ""]
    out["labels"] = out["label_str"].map(LABEL2ID).astype(int)
    print(f"[{name}] {n0 - len(out)} baris dibuang (label '-'/tidak valid atau teks kosong), "
          f"sisa {len(out)}")
    return out.reset_index(drop=True)


def make_triplet_tokenizer(tokenizer, max_length):
    """Tokenizer bawaan hanya mendukung 1 atau 2 kalimat, jadi 3 segmen dirakit manual
    mengikuti format pasangan kalimat XLM-R:
        <s> P </s></s> H </s></s> E </s>
    Jika melebihi max_length, segmen yang paling panjang dipotong lebih dulu
    (setara strategi 'longest_first')."""
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    n_special = 6  # <s>, </s></s>, </s></s>, </s>
    budget = max_length - n_special

    def tokenize(batch):
        seg_ids = {c: tokenizer(batch[c], add_special_tokens=False)["input_ids"] for c in SEGMENTS}
        input_ids, attention_mask = [], []
        for p, h, e in zip(*(seg_ids[c] for c in SEGMENTS)):
            segs = [list(p), list(h), list(e)]
            while sum(len(s) for s in segs) > budget:
                longest = max(segs, key=len)
                longest.pop()
            ids = ([cls_id] + segs[0] + [sep_id, sep_id] + segs[1]
                   + [sep_id, sep_id] + segs[2] + [sep_id])
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return tokenize


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
    p.add_argument("--explanation_col", default="penjelasan",
                   help="Kolom penjelasan untuk training.")
    p.add_argument("--eval_explanation_cols", nargs="*", default=None,
                   help="Kolom penjelasan untuk evaluasi dev/test, mis. penjelasan_1 penjelasan_2 "
                        "penjelasan_3. Default: sama dengan --explanation_col.")
    p.add_argument("--model_name", default="xlm-roberta-base")
    p.add_argument("--output_dir", default="outputs/phe_to_label_xlmr")
    p.add_argument("--max_length", type=int, default=192,
                   help="Panjang maksimal gabungan premis + hipotesis + penjelasan (token).")
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
    eval_cols = args.eval_explanation_cols or [args.explanation_col]
    base_cols = (args.label_col, args.premise_col, args.hypothesis_col)

    # --- Data ---
    raw = {
        "train": load_jsonl(args.train_file),
        "validation": load_jsonl(args.dev_file),
        "test": load_jsonl(args.test_file),
    }
    train_df = clean_split(raw["train"], *base_cols, args.explanation_col, "train")
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)),
                                   random_state=args.seed).reset_index(drop=True)
    val_df = clean_split(raw["validation"], *base_cols, eval_cols[0], "validation")
    print("Distribusi label train:\n", train_df["label_str"].value_counts())

    # --- Tokenisasi 3 segmen (padding dinamis) ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenize = make_triplet_tokenizer(tokenizer, args.max_length)

    def to_ds(df):
        return Dataset.from_pandas(df[SEGMENTS + ["labels"]]).map(
            tokenize, batched=True, remove_columns=SEGMENTS)

    dataset = DatasetDict({"train": to_ds(train_df), "validation": to_ds(val_df)})
    print(dataset)
    print("Contoh input ter-tokenisasi:", tokenizer.decode(dataset["train"][0]["input_ids"]))
    lens = [len(x) for x in dataset["train"]["input_ids"]]
    print(f"Panjang token train: rata-rata {np.mean(lens):.1f}, maks {max(lens)}, "
          f"terpotong {sum(l >= args.max_length for l in lens)} contoh")

    # --- Model & Trainer ---
    model = XLMRobertaPHEClassifier(args.model_name, num_labels=len(LABEL_NAMES))

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

    best_path = os.path.join(args.output_dir, "best_phe_to_label_xlmr_model.pt")
    torch.save(trainer.model.state_dict(), best_path)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Bobot model terbaik disimpan di {best_path}")

    # --- Evaluasi validasi & test ---
    results = {"config": vars(args)}
    labels_idx = list(range(len(LABEL_NAMES)))
    for split in ["validation", "test"]:
        results[split] = {}
        for col in eval_cols:
            if col not in raw[split].columns:
                print(f"[{split}] kolom '{col}' tidak ada, dilewati.")
                continue
            df = clean_split(raw[split], *base_cols, col, f"{split}/{col}")
            out = trainer.predict(to_ds(df))
            y_pred = out.predictions.argmax(-1)
            y_true = df["labels"].values

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

            print("\n" + "=" * 60)
            print(f"   Laporan Kinerja — {split.upper()} (kalimat1 + kalimat2 + {col})")
            print("=" * 60)
            print(f"Akurasi         : {acc:.4f}")
            print(f"F1-Score (Macro): {f1:.4f}")
            print("-" * 60)
            print(classification_report(y_true, y_pred, labels=labels_idx,
                                        target_names=LABEL_NAMES, digits=4, zero_division=0))
            print("Confusion matrix (baris=gold, kolom=prediksi):", LABEL_NAMES)
            print(cm)

            results[split][col] = {
                "accuracy": float(acc), "macro_f1": float(f1),
                "per_class": classification_report(
                    y_true, y_pred, labels=labels_idx, target_names=LABEL_NAMES,
                    output_dict=True, zero_division=0),
                "confusion_matrix": cm.tolist(),
            }

            pred_df = df[SEGMENTS + ["label_str"]].copy()
            pred_df["prediksi"] = [ID2LABEL[i] for i in y_pred]
            probs = torch.softmax(torch.tensor(out.predictions), dim=-1).numpy()
            for i, lab in ID2LABEL.items():
                pred_df[f"prob_{lab}"] = probs[:, i]
            pred_df.to_csv(os.path.join(args.output_dir, f"pred_{split}_{col}.csv"), index=False)

        if len(results[split]) > 1:
            accs = [v["accuracy"] for v in results[split].values()]
            f1s = [v["macro_f1"] for v in results[split].values()]
            results[split]["mean_over_explanations"] = {
                "accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            }
            print(f"\n[{split}] rata-rata: acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
                  f"macro-F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSelesai. Hasil di {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
