#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e-SNLI Indonesia — Generasi penjelasan:
    kalimat1 (premis) + kalimat2 (hipotesis) + label emas -> penjelasan
Model : Gemma kecil (decoder-only), default google/gemma-3-270m, fine-tune dengan LoRA
        (atau full fine-tuning dengan --full_finetune).

Format prompt (sama untuk training & inferensi):
    Premis: <kalimat1>
    Hipotesis: <kalimat2>
    Label: <keterlibatan | netral | kontradiksi>
    Penjelasan: <penjelasan><eos>
Saat inferensi, label yang dimasukkan adalah label emas (gold label) dari data.
Loss hanya dihitung pada bagian penjelasan (token prompt di-mask -100).

Evaluasi (validation & test): BLEU (sacrebleu, multi-referensi jika ada beberapa kolom
penjelasan), ROUGE-L F1 (maks. terhadap semua referensi), dan file CSV hasil generasi.

Dependensi:
    pip install "transformers>=4.50" datasets accelerate peft sacrebleu rouge_score pandas torch

Model Gemma bersifat "gated": setujui lisensinya di halaman model Hugging Face, lalu
    huggingface-cli login     (atau: export HF_TOKEN=hf_xxx)

Contoh:
    python train_phl_to_expl_gemma.py \
        --train_file esnli_indo/train_data.jsonl \
        --dev_file   esnli_indo/validation_data.jsonl \
        --test_file  esnli_indo/test_data.jsonl \
        --model_name google/gemma-3-270m
"""

import argparse
import inspect
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from rouge_score import rouge_scorer
import sacrebleu
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

VALID_LABELS = {"keterlibatan", "netral", "kontradiksi"}


def build_prompt(premise: str, hypothesis: str, label: str) -> str:
    return f"Premis: {premise}\nHipotesis: {hypothesis}\nLabel: {label}\nPenjelasan:"


# --------------------------------------------------------------------------- #
# Utilitas
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


def clean_split(df, prem_col, hyp_col, expl_cols, label_col, name):
    """Buang baris berlabel '-' dan baris dengan premis / hipotesis / semua penjelasan kosong.
    Mengembalikan kolom: premise, hypothesis, refs (list penjelasan), label_str."""
    for col in [prem_col, hyp_col] + expl_cols:
        if col not in df.columns:
            raise KeyError(f"[{name}] kolom '{col}' tidak ada. Kolom: {list(df.columns)}")
    n0 = len(df)
    df = df.copy()
    if label_col not in df.columns:
        raise KeyError(f"[{name}] kolom label '{label_col}' tidak ada. Kolom: {list(df.columns)}")
    df["label_str"] = df[label_col].astype(str).str.strip().str.lower()
    df = df[df["label_str"].isin(VALID_LABELS)]
    df["premise"] = df[prem_col].fillna("").astype(str).str.strip()
    df["hypothesis"] = df[hyp_col].fillna("").astype(str).str.strip()

    def refs_of(row):
        return [str(row[c]).strip() for c in expl_cols
                if pd.notna(row[c]) and str(row[c]).strip()]

    df["refs"] = df.apply(refs_of, axis=1)
    df = df[(df["premise"] != "") & (df["hypothesis"] != "") & (df["refs"].str.len() > 0)]
    print(f"[{name}] {n0 - len(df)} baris dibuang (label '-' atau teks kosong), sisa {len(df)}")
    return df[["premise", "hypothesis", "refs", "label_str"]].reset_index(drop=True)


def make_train_tokenizer(tokenizer, max_length):
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos = [tokenizer.eos_token_id]

    def tokenize(batch):
        input_ids, labels = [], []
        for prem, hyp, lab, expl in zip(batch["premise"], batch["hypothesis"],
                                        batch["label_str"], batch["target"]):
            p_ids = bos + tokenizer(build_prompt(prem, hyp, lab),
                                    add_special_tokens=False)["input_ids"]
            t_ids = tokenizer(" " + expl, add_special_tokens=False)["input_ids"] + eos
            ids = (p_ids + t_ids)[:max_length]
            lab = ([-100] * len(p_ids) + t_ids)[:max_length]
            input_ids.append(ids)
            labels.append(lab)
        return {"input_ids": input_ids, "labels": labels}

    return tokenize


class CausalLMCollator:
    """Padding kanan: input_ids dengan pad_token_id, labels dengan -100."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        L = max(len(f["input_ids"]) for f in features)
        ids, labs, mask = [], [], []
        for f in features:
            n = len(f["input_ids"])
            ids.append(list(f["input_ids"]) + [self.pad_id] * (L - n))
            labs.append(list(f["labels"]) + [-100] * (L - n))
            mask.append([1] * n + [0] * (L - n))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


@torch.no_grad()
def generate(model, tokenizer, premises, hypotheses, labels, batch_size, max_new_tokens,
             num_beams):
    """Generasi batch dengan padding kiri; ambil hanya token baru, potong di baris baru."""
    model.eval()
    device = next(model.parameters()).device
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    outs = []
    for i in range(0, len(hypotheses), batch_size):
        chunk = zip(premises[i:i + batch_size], hypotheses[i:i + batch_size],
                    labels[i:i + batch_size])
        prompts = [bos + tokenizer(build_prompt(p, h, l), add_special_tokens=False)["input_ids"]
                   for p, h, l in chunk]
        L = max(len(p) for p in prompts)
        ids = torch.tensor([[tokenizer.pad_token_id] * (L - len(p)) + p for p in prompts],
                           device=device)
        mask = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in prompts], device=device)
        gen = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        for row in gen[:, L:]:
            text = tokenizer.decode(row, skip_special_tokens=True)
            outs.append(text.split("\n")[0].strip())
        print(f"  generasi {min(i + batch_size, len(hypotheses))}/{len(hypotheses)}", flush=True)
    return outs


def text_metrics(preds, refs_list):
    """BLEU korpus (multi-referensi) + ROUGE-L F1 (maks. antar referensi, dirata-rata)."""
    n_ref = max(len(r) for r in refs_list)
    # sacrebleu: list per posisi referensi; referensi yang kurang diisi dengan ref pertama
    ref_streams = [[r[k] if k < len(r) else r[0] for r in refs_list] for k in range(n_ref)]
    bleu = sacrebleu.corpus_bleu(preds, ref_streams).score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = [max(scorer.score(ref, pred)["rougeL"].fmeasure for ref in refs)
               for pred, refs in zip(preds, refs_list)]
    return {"bleu": float(bleu), "rougeL_f1": float(np.mean(rouge_l) * 100),
            "num_references": n_ref}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default="esnli_indo/train_data.jsonl")
    p.add_argument("--dev_file", default="esnli_indo/validation_data.jsonl")
    p.add_argument("--test_file", default="esnli_indo/test_data.jsonl")
    p.add_argument("--premise_col", default="kalimat1")
    p.add_argument("--hypothesis_col", default="kalimat2")
    p.add_argument("--explanation_col", default="penjelasan",
                   help="Kolom penjelasan (target) untuk training.")
    p.add_argument("--eval_explanation_cols", nargs="*", default=None,
                   help="Kolom penjelasan referensi untuk evaluasi, mis. penjelasan_1 "
                        "penjelasan_2 penjelasan_3. Default: --explanation_col.")
    p.add_argument("--label_col", default="label emas",
                   help="Kolom label emas; dimasukkan ke prompt (baris berlabel '-' dibuang).")
    p.add_argument("--model_name", default="google/gemma-3-270m")
    p.add_argument("--output_dir", default="outputs/phl_to_expl_gemma")
    p.add_argument("--full_finetune", action="store_true",
                   help="Latih semua bobot (default: LoRA).")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=192,
                   help="Maks token prompt (premis + hipotesis + label) + penjelasan.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None,
                   help="Default: 2e-4 untuk LoRA, 5e-5 untuk full fine-tuning.")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--patience", type=int, default=3, help="Early stopping (per evaluasi); 0 = off")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_loss_samples", type=int, default=2000,
                   help="Jumlah contoh validasi untuk eval loss selama training (biar cepat).")
    p.add_argument("--max_gen_samples", type=int, default=None,
                   help="Batasi jumlah contoh yang digenerasi per split (default: semua).")
    p.add_argument("--gen_batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    eval_cols = args.eval_explanation_cols or [args.explanation_col]
    lr = args.lr or (5e-5 if args.full_finetune else 2e-4)

    # --- Data ---
    raw = {"train": load_jsonl(args.train_file),
           "validation": load_jsonl(args.dev_file),
           "test": load_jsonl(args.test_file)}
    train_df = clean_split(raw["train"], args.premise_col, args.hypothesis_col, [args.explanation_col],
                           args.label_col, "train")
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)),
                                   random_state=args.seed).reset_index(drop=True)
    val_df = clean_split(raw["validation"], args.premise_col, args.hypothesis_col, eval_cols,
                         args.label_col, "validation")

    # --- Tokenizer & model ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    # transformers >= 5 memakai `dtype`, versi 4.x memakai `torch_dtype`
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="eager",   # disarankan untuk training Gemma 3
        **{dtype_key: dtype},
    )
    model.config.use_cache = False

    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # --- Dataset training (target = penjelasan pertama) ---
    tokenize = make_train_tokenizer(tokenizer, args.max_length)

    def to_train_ds(df):
        d = pd.DataFrame({"premise": df["premise"], "hypothesis": df["hypothesis"],
                          "label_str": df["label_str"], "target": df["refs"].str[0]})
        return Dataset.from_pandas(d).map(
            tokenize, batched=True,
            remove_columns=["premise", "hypothesis", "label_str", "target"])

    train_ds = to_train_ds(train_df)
    eval_loss_df = val_df.sample(n=min(args.max_eval_loss_samples, len(val_df)),
                                 random_state=args.seed)
    eval_ds = to_train_ds(eval_loss_df)
    ex = train_ds[0]
    print("Contoh input :", tokenizer.decode(ex["input_ids"]))
    print("Contoh target:", tokenizer.decode([t for t in ex["labels"] if t != -100]))

    # --- Trainer ---
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    warmup_kw = ({"warmup_ratio": args.warmup_ratio} if "warmup_ratio" in ta_params
                 else {"warmup_steps": args.warmup_ratio})
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        **warmup_kw,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        bf16=use_bf16,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(args.patience)] if args.patience > 0 else [],
    )
    print(f"Memulai pelatihan di device: {training_args.device} | "
          f"{'full fine-tuning' if args.full_finetune else 'LoRA'} | lr={lr}")
    trainer.train()

    best_dir = os.path.join(args.output_dir, "best_model")
    trainer.model.save_pretrained(best_dir)   # LoRA: hanya adapter; full: seluruh model
    tokenizer.save_pretrained(best_dir)
    print(f"Model terbaik disimpan di {best_dir}")

    # --- Generasi & evaluasi ---
    model = trainer.model
    model.config.use_cache = True
    results = {"config": {**vars(args), "lr": lr}}
    for split in ["validation", "test"]:
        cols = [c for c in eval_cols if c in raw[split].columns]
        if not cols:
            print(f"[{split}] tidak ada kolom referensi, dilewati.")
            continue
        df = clean_split(raw[split], args.premise_col, args.hypothesis_col, cols, args.label_col, split)
        if args.max_gen_samples:
            df = df.head(args.max_gen_samples)
        print(f"\nGenerasi penjelasan untuk {split} ({len(df)} contoh)...")
        preds = generate(model, tokenizer, df["premise"].tolist(), df["hypothesis"].tolist(),
                         df["label_str"].tolist(), args.gen_batch_size,
                         args.max_new_tokens, args.num_beams)
        m = text_metrics(preds, df["refs"].tolist())
        results[split] = m

        print("=" * 55)
        print(f"   Hasil Generasi — {split.upper()}")
        print("=" * 55)
        print(f"BLEU       : {m['bleu']:.2f}  ({m['num_references']} referensi)")
        print(f"ROUGE-L F1 : {m['rougeL_f1']:.2f}")
        rows = zip(df["premise"], df["hypothesis"], df["label_str"], df["refs"], preds)
        for pr, h, lab, r, p in list(rows)[:5]:
            print(f"- Premis    : {pr}\n  Hipotesis : {h}\n  Label     : {lab}\n"
                  f"  Referensi : {r[0]}\n  Prediksi  : {p}")

        out_df = pd.DataFrame({
            "kalimat1": df["premise"],
            "kalimat2": df["hypothesis"],
            "label_emas": df["label_str"],
            "penjelasan_generasi": preds,
        })
        for k, c in enumerate(cols):
            out_df[f"ref_{c}"] = df["refs"].apply(lambda r, k=k: r[k] if k < len(r) else "")
        out_df.to_csv(os.path.join(args.output_dir, f"gen_{split}.csv"), index=False)

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSelesai. Hasil di {args.output_dir}/results.json")


if __name__ == "__main__":
    main()