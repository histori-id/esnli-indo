#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e-SNLI Indonesia — Pipeline PREDICT-THEN-EXPLAIN (inferensi saja, tanpa training ulang)

    kalimat1 + kalimat2 ──► XLM-R classifier ──► label prediksi
                                                     │
    kalimat1 + kalimat2 + label prediksi ──► Gemma generator ──► penjelasan
                                                     │
    (opsional) penjelasan ──► XLM-R expl->label ──► cek konsistensi

Model yang dipakai (hasil skrip training sebelumnya):
    1. Classifier P+H -> label     : train_ph_to_label_xlmr.py   (outputs/ph_to_label_xlmr)
    2. Generator P+H+label -> expl : train_phl_to_expl_gemma.py  (outputs/phl_to_expl_gemma/best_model)
    3. (opsional) expl -> label    : train_expl_to_label_xlmr.py (outputs/expl_to_label_xlmr)

Skrip ini mengimpor kelas model & fungsi dari ketiga skrip di atas, jadi letakkan semuanya
di folder yang sama.

Evaluasi:
    - Akurasi & macro-F1 label prediksi vs label emas
    - BLEU & ROUGE-L penjelasan: semua contoh, contoh dengan label benar, dan label salah
    - Konsistensi: label dari classifier expl->label vs label prediksi (dan vs label emas)
    - (opsional, --also_gold_label) generasi dengan label emas sebagai batas atas

Contoh:
    python predict_then_explain.py \
        --test_file esnli_indo/test_data.jsonl \
        --clf_dir outputs/ph_to_label_xlmr \
        --gen_dir outputs/phl_to_expl_gemma/best_model \
        --expl_clf_dir outputs/expl_to_label_xlmr
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ph_to_label_xlmr import XLMRobertaPHClassifier, LABEL_NAMES, ID2LABEL  # noqa: E402
from train_expl_to_label_xlmr import XLMRobertaExplClassifier  # noqa: E402
from train_phl_to_expl_gemma import (  # noqa: E402
    clean_split,
    generate as gemma_generate,
    text_metrics,
)


# --------------------------------------------------------------------------- #
# Utilitas memuat model
# --------------------------------------------------------------------------- #
def read_train_config(model_dir):
    """Ambil konfigurasi training (model_name, max_length) dari results.json, jika ada."""
    path = os.path.join(model_dir, "results.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("config", {})
    return {}


def find_checkpoint(model_dir, filename):
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    return path


def load_xlmr_classifier(cls, model_dir, ckpt_name, base_model, device):
    cfg = read_train_config(model_dir)
    model_name = base_model or cfg.get("model_name", "xlm-roberta-base")
    max_len = cfg.get("max_length", 128)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = cls(model_name, num_labels=len(LABEL_NAMES))
    state = torch.load(find_checkpoint(model_dir, ckpt_name), map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"✅ Classifier dimuat dari {model_dir} (base: {model_name}, max_length: {max_len})")
    return model, tokenizer, max_len


def load_gemma(gen_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(gen_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    import transformers
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

    if os.path.exists(os.path.join(gen_dir, "adapter_config.json")):
        from peft import PeftConfig, PeftModel
        base_name = PeftConfig.from_pretrained(gen_dir).base_model_name_or_path
        base = AutoModelForCausalLM.from_pretrained(base_name, **{dtype_key: dtype})
        model = PeftModel.from_pretrained(base, gen_dir)
        model = model.merge_and_unload()           # gabungkan LoRA -> inferensi lebih cepat
        print(f"✅ Generator dimuat: base {base_name} + adapter LoRA {gen_dir}")
    else:
        model = AutoModelForCausalLM.from_pretrained(gen_dir, **{dtype_key: dtype})
        print(f"✅ Generator dimuat (full model) dari {gen_dir}")
    model.config.use_cache = True
    model.to(device).eval()
    return model, tokenizer


def free_gpu():
    """Panggil setelah `del model` agar memori GPU dilepas sebelum model berikutnya dimuat."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Inferensi classifier
# --------------------------------------------------------------------------- #
@torch.no_grad()
def classify(model, tokenizer, texts_a, texts_b, max_len, batch_size, device):
    """texts_b=None -> input satu kalimat (penjelasan); selain itu sentence pair (P, H)."""
    collator = DataCollatorWithPadding(tokenizer)
    preds, probs = [], []
    for i in range(0, len(texts_a), batch_size):
        a = texts_a[i:i + batch_size]
        b = texts_b[i:i + batch_size] if texts_b is not None else None
        enc = tokenizer(a, b, truncation=True, max_length=max_len) if b is not None \
            else tokenizer(a, truncation=True, max_length=max_len)
        feats = [{k: enc[k][j] for k in ("input_ids", "attention_mask")} for j in range(len(a))]
        batch = {k: v.to(device) for k, v in collator(feats).items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probs.append(p)
        preds.extend(p.argmax(-1).tolist())
    return np.array(preds), np.concatenate(probs)


def label_report(y_true, y_pred, title):
    idx = list(range(len(LABEL_NAMES)))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=idx)
    print("\n" + "=" * 60 + f"\n   {title}\n" + "=" * 60)
    print(f"Akurasi         : {acc:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    print(classification_report(y_true, y_pred, labels=idx, target_names=LABEL_NAMES,
                                digits=4, zero_division=0))
    print("Confusion matrix (baris=acuan, kolom=prediksi):", LABEL_NAMES)
    print(cm)
    return {"accuracy": float(acc), "macro_f1": float(f1), "confusion_matrix": cm.tolist(),
            "per_class": classification_report(y_true, y_pred, labels=idx,
                                               target_names=LABEL_NAMES, output_dict=True,
                                               zero_division=0)}


def subset_text_metrics(preds, refs, mask, name):
    preds = [p for p, m in zip(preds, mask) if m]
    refs = [r for r, m in zip(refs, mask) if m]
    if not preds:
        print(f"  {name}: tidak ada contoh")
        return {"n": 0}
    m = text_metrics(preds, refs)
    m["n"] = len(preds)
    print(f"  {name:<22} n={m['n']:<6} BLEU={m['bleu']:.2f}  ROUGE-L={m['rougeL_f1']:.2f}")
    return m


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", default="dataset/gemini/test.jsonl")
    p.add_argument("--premise_col", default="Sentence1")
    p.add_argument("--hypothesis_col", default="Sentence2")
    p.add_argument("--label_col", default="gold_label")
    p.add_argument("--explanation_cols", nargs="+", default=["Explanation_1"],
                   help="Kolom penjelasan referensi, mis. penjelasan_1 penjelasan_2 penjelasan_3")
    p.add_argument("--clf_dir", default="outputs/ph_to_label_xlmr")
    p.add_argument("--clf_ckpt", default="best_ph_to_label_xlmr_model.pt")
    p.add_argument("--gen_dir", default="outputs/phl_to_expl_gemma/best_model")
    p.add_argument("--expl_clf_dir", default="outputs/expl_to_label_xlmr",
                   help="Classifier expl->label untuk cek konsistensi. Kosongkan ('') untuk skip.")
    p.add_argument("--expl_clf_ckpt", default="best_expl_to_label_xlmr_model.pt")
    p.add_argument("--base_model", default=None,
                   help="Nama base XLM-R jika results.json tidak ada (default xlm-roberta-base).")
    p.add_argument("--also_gold_label", action="store_true",
                   help="Generasi juga dengan label emas (batas atas kualitas penjelasan).")
    p.add_argument("--output_dir", default="outputs/predict_then_explain")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--clf_batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_beams", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    raw = pd.read_json(args.test_file, lines=True)
    cols = [c for c in args.explanation_cols if c in raw.columns]
    if not cols:
        raise KeyError(f"Tidak ada kolom penjelasan {args.explanation_cols} di {args.test_file}")
    df = clean_split(raw, args.premise_col, args.hypothesis_col, cols, args.label_col, "test")
    if args.max_samples:
        df = df.head(args.max_samples).reset_index(drop=True)
    premises, hyps = df["premise"].tolist(), df["hypothesis"].tolist()
    refs = df["refs"].tolist()
    label2id = {l: i for i, l in enumerate(LABEL_NAMES)}
    y_gold = df["label_str"].map(label2id).values
    results = {"config": vars(args), "n": len(df)}

    # --- Tahap 1: PREDICT (XLM-R P+H -> label) ---
    clf, clf_tok, clf_len = load_xlmr_classifier(
        XLMRobertaPHClassifier, args.clf_dir, args.clf_ckpt, args.base_model, device)
    y_pred, p_pred = classify(clf, clf_tok, premises, hyps, clf_len, args.clf_batch_size, device)
    del clf
    free_gpu()
    pred_labels = [ID2LABEL[i] for i in y_pred]
    results["label"] = label_report(y_gold, y_pred, "TAHAP 1 — Prediksi label (vs label emas)")
    correct = (y_pred == y_gold)

    # --- Tahap 2: EXPLAIN (Gemma P+H+label prediksi -> penjelasan) ---
    gen, gen_tok = load_gemma(args.gen_dir, device)
    print(f"\nGenerasi penjelasan dengan LABEL PREDIKSI ({len(df)} contoh)...")
    expl_pred = gemma_generate(gen, gen_tok, premises, hyps, pred_labels,
                               args.gen_batch_size, args.max_new_tokens, args.num_beams)
    expl_gold = None
    if args.also_gold_label:
        print("\nGenerasi penjelasan dengan LABEL EMAS (batas atas)...")
        expl_gold = gemma_generate(gen, gen_tok, premises, hyps, df["label_str"].tolist(),
                                   args.gen_batch_size, args.max_new_tokens, args.num_beams)
    del gen
    free_gpu()

    print("\n" + "=" * 60 + "\n   TAHAP 2 — Kualitas penjelasan (label prediksi)\n" + "=" * 60)
    results["explanation"] = {
        "all": subset_text_metrics(expl_pred, refs, [True] * len(df), "semua"),
        "label_benar": subset_text_metrics(expl_pred, refs, correct, "label prediksi benar"),
        "label_salah": subset_text_metrics(expl_pred, refs, ~correct, "label prediksi salah"),
    }
    if expl_gold is not None:
        print("  --- batas atas (label emas) ---")
        results["explanation_gold_label"] = subset_text_metrics(
            expl_gold, refs, [True] * len(df), "semua (label emas)")

    # --- Tahap 3 (opsional): konsistensi penjelasan vs label ---
    expl_lab = None
    if args.expl_clf_dir:
        try:
            eclf, etok, elen = load_xlmr_classifier(
                XLMRobertaExplClassifier, args.expl_clf_dir, args.expl_clf_ckpt,
                args.base_model, device)
        except FileNotFoundError as e:
            print(f"⚠️  Cek konsistensi dilewati: {e}")
            eclf = None
        if eclf is not None:
            e_pred, _ = classify(eclf, etok, expl_pred, None, elen, args.clf_batch_size, device)
            del eclf
            free_gpu()
            expl_lab = [ID2LABEL[i] for i in e_pred]
            cons_pred = float(np.mean(e_pred == y_pred))
            cons_gold = float(np.mean(e_pred == y_gold))
            print("\n" + "=" * 60 + "\n   TAHAP 3 — Konsistensi penjelasan\n" + "=" * 60)
            print(f"Label dari penjelasan == label prediksi : {cons_pred:.4f}")
            print(f"Label dari penjelasan == label emas     : {cons_gold:.4f}")
            results["consistency"] = {
                "expl_label_vs_pred_label": cons_pred,
                "expl_label_vs_gold_label": cons_gold,
                "vs_pred_report": label_report(
                    y_pred, e_pred, "Konsistensi: label dari penjelasan vs label prediksi"),
            }

    # --- Simpan ---
    out = pd.DataFrame({
        "kalimat1": df["premise"], "kalimat2": df["hypothesis"],
        "label_emas": df["label_str"], "label_prediksi": pred_labels,
        "label_benar": correct,
    })
    for i, lab in ID2LABEL.items():
        out[f"prob_{lab}"] = p_pred[:, i]
    out["penjelasan_generasi"] = expl_pred
    if expl_gold is not None:
        out["penjelasan_dgn_label_emas"] = expl_gold
    if expl_lab is not None:
        out["label_dari_penjelasan"] = expl_lab
    for k, c in enumerate(cols):
        out[f"ref_{c}"] = df["refs"].apply(lambda r, k=k: r[k] if k < len(r) else "")
    csv_path = os.path.join(args.output_dir, "predict_then_explain_test.csv")
    out.to_csv(csv_path, index=False)

    print("\nContoh hasil:")
    for _, r in out.head(5).iterrows():
        print(f"- P: {r['kalimat1']}\n  H: {r['kalimat2']}\n"
              f"  Emas: {r['label_emas']} | Prediksi: {r['label_prediksi']}\n"
              f"  Penjelasan: {r['penjelasan_generasi']}")

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSelesai. CSV: {csv_path}\nMetrik: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
