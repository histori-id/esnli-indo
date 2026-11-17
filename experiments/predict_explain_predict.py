#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e-SNLI Indonesia — Pipeline PREDICT -> EXPLAIN -> PREDICT AGAIN (inferensi saja)

    Tahap 1  kalimat1 + kalimat2 ──► XLM-R (P+H->label) ──► label awal
    Tahap 2  kalimat1 + kalimat2 + label awal ──► Gemma ──► penjelasan
    Tahap 3  penjelasan (± P+H) ──► XLM-R classifier ──► LABEL AKHIR

Berbeda dengan predict_then_explain.py (tahap 3 hanya untuk evaluasi konsistensi),
di sini label dari tahap 3 DIPAKAI sebagai jawaban akhir.

Pilihan input tahap 3 (--final_input):
    expl : penjelasan saja
           model: train_expl_to_label_xlmr.py      -> outputs/expl_to_label_xlmr
    dual : [P+H] dan [penjelasan] di-encode terpisah, lalu digabung (dual encoder)
           model: train_ph_e_dual_xlmr.py           -> outputs/ph_e_dual_xlmr
    phe  : P + H + penjelasan dalam SATU urutan
           model: train_phe_to_label_xlmr.py (versi 3 segmen) -> outputs/phe_to_label_xlmr

Kelas model diimpor dari skrip training; hanya skrip untuk mode yang dipakai yang perlu ada.
Untuk mode dual, kelas dicari di train_ph_e_dual_xlmr.py lalu train_phe_to_label_xlmr.py.

Yang dilaporkan:
    - Akurasi label awal (tahap 1) dan label akhir (tahap 3) vs label emas
    - Perubahan label: berapa yang berubah, benar->salah, salah->benar
    - Prediksi selektif: hanya contoh yang label tahap 1 == tahap 3 (coverage & akurasi)
    - BLEU & ROUGE-L penjelasan

Contoh:
    python predict_explain_predict.py --final_input dual
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_then_explain import (  # noqa: E402
    classify,
    find_checkpoint,
    free_gpu,
    label_report,
    load_gemma,
    load_xlmr_classifier,
    read_train_config,
    subset_text_metrics,
)
from train_ph_to_label_xlmr import XLMRobertaPHClassifier, LABEL_NAMES, ID2LABEL  # noqa: E402
from train_phl_to_expl_gemma import clean_split, generate as gemma_generate  # noqa: E402

FINAL_DEFAULTS = {
    "expl": ("outputs/expl_to_label_xlmr", "best_expl_to_label_xlmr_model.pt"),
    "dual": ("outputs/ph_e_dual_xlmr", "best_ph_e_dual_xlmr_model.pt"),
    "phe": ("outputs/phe_to_label_xlmr", "best_phe_to_label_xlmr_model.pt"),
}


def import_attr(attr, modules):
    """Impor `attr` dari modul pertama yang memilikinya."""
    for m in modules:
        try:
            mod = importlib.import_module(m)
        except ImportError:
            continue
        if hasattr(mod, attr):
            return getattr(mod, attr)
    raise ImportError(f"'{attr}' tidak ditemukan di {modules}. Pastikan skrip training yang "
                      f"sesuai ada di folder yang sama dengan skrip ini.")


# --------------------------------------------------------------------------- #
# Tahap 3: loader & inferensi per mode
# --------------------------------------------------------------------------- #
def load_dual_classifier(model_dir, ckpt_name, base_model, device):
    cls = import_attr("XLMRobertaDualClassifier",
                      ["train_ph_e_dual_xlmr", "train_phe_to_label_xlmr"])
    cfg = read_train_config(model_dir)
    model_name = base_model or cfg.get("model_name", "xlm-roberta-base")
    model = cls(model_name, num_labels=len(LABEL_NAMES),
                fusion=cfg.get("fusion", "concat"),
                separate_encoders=cfg.get("separate_encoders", False))
    model.load_state_dict(torch.load(find_checkpoint(model_dir, ckpt_name), map_location="cpu"))
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    lens = (cfg.get("max_len_ph", 128), cfg.get("max_len_e", 64))
    print(f"✅ Dual classifier dimuat dari {model_dir} (base: {model_name}, "
          f"fusi: {cfg.get('fusion', 'concat')}, encoder terpisah: "
          f"{cfg.get('separate_encoders', False)})")
    return model, tokenizer, lens


@torch.no_grad()
def classify_dual(model, tokenizer, premises, hyps, expls, lens, batch_size, device):
    len_ph, len_e = lens
    preds, probs = [], []
    for i in range(0, len(premises), batch_size):
        enc_ph = tokenizer(premises[i:i + batch_size], hyps[i:i + batch_size],
                           truncation=True, max_length=len_ph)
        enc_e = tokenizer(expls[i:i + batch_size], truncation=True, max_length=len_e)
        pad = lambda enc: tokenizer.pad(  # noqa: E731
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
            return_tensors="pt")
        b_ph, b_e = pad(enc_ph), pad(enc_e)
        logits = model(input_ids_ph=b_ph["input_ids"].to(device),
                       attention_mask_ph=b_ph["attention_mask"].to(device),
                       input_ids_e=b_e["input_ids"].to(device),
                       attention_mask_e=b_e["attention_mask"].to(device)).logits
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probs.append(p)
        preds.extend(p.argmax(-1).tolist())
    return np.array(preds), np.concatenate(probs)


@torch.no_grad()
def classify_phe(model, tokenizer, premises, hyps, expls, max_len, batch_size, device):
    """Klasifikasi 3 segmen <s> P </s></s> H </s></s> E </s> (sama dengan training phe)."""
    make_triplet_tokenizer = import_attr("make_triplet_tokenizer", ["train_phe_to_label_xlmr"])
    tok3 = make_triplet_tokenizer(tokenizer, max_len)
    collator = DataCollatorWithPadding(tokenizer)
    preds, probs = [], []
    for i in range(0, len(premises), batch_size):
        enc = tok3({"premise": premises[i:i + batch_size],
                    "hypothesis": hyps[i:i + batch_size],
                    "explanation": expls[i:i + batch_size]})
        feats = [{"input_ids": a, "attention_mask": m}
                 for a, m in zip(enc["input_ids"], enc["attention_mask"])]
        batch = {k: v.to(device) for k, v in collator(feats).items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probs.append(p)
        preds.extend(p.argmax(-1).tolist())
    return np.array(preds), np.concatenate(probs)


def run_final_stage(mode, final_dir, final_ckpt, base_model, premises, hyps, expls,
                    batch_size, device):
    if mode == "expl":
        cls = import_attr("XLMRobertaExplClassifier", ["train_expl_to_label_xlmr"])
        model, tok, max_len = load_xlmr_classifier(cls, final_dir, final_ckpt, base_model, device)
        out = classify(model, tok, expls, None, max_len, batch_size, device)
    elif mode == "dual":
        model, tok, lens = load_dual_classifier(final_dir, final_ckpt, base_model, device)
        out = classify_dual(model, tok, premises, hyps, expls, lens, batch_size, device)
    else:  # phe
        cls = import_attr("XLMRobertaPHEClassifier", ["train_phe_to_label_xlmr"])
        model, tok, _ = load_xlmr_classifier(cls, final_dir, final_ckpt, base_model, device)
        max_len = read_train_config(final_dir).get("max_length", 192)
        out = classify_phe(model, tok, premises, hyps, expls, max_len, batch_size, device)
    del model
    free_gpu()
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", default="esnli_indo/test_data.jsonl")
    p.add_argument("--premise_col", default="kalimat1")
    p.add_argument("--hypothesis_col", default="kalimat2")
    p.add_argument("--label_col", default="label emas")
    p.add_argument("--explanation_cols", nargs="+", default=["penjelasan"],
                   help="Kolom penjelasan referensi, mis. penjelasan_1 penjelasan_2 penjelasan_3")
    p.add_argument("--clf_dir", default="outputs/ph_to_label_xlmr")
    p.add_argument("--clf_ckpt", default="best_ph_to_label_xlmr_model.pt")
    p.add_argument("--gen_dir", default="outputs/phl_to_expl_gemma/best_model")
    p.add_argument("--final_input", choices=["expl", "dual", "phe"], default="expl",
                   help="Input classifier tahap 3: penjelasan saja, dual encoder [P+H]&[E], "
                        "atau P+H+E dalam satu urutan.")
    p.add_argument("--final_clf_dir", default=None,
                   help="Default sesuai mode (lihat docstring).")
    p.add_argument("--final_clf_ckpt", default=None)
    p.add_argument("--base_model", default=None,
                   help="Nama base XLM-R jika results.json tidak ada (default xlm-roberta-base).")
    p.add_argument("--output_dir", default=None,
                   help="Default: outputs/predict_explain_predict_<final_input>")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--clf_batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_beams", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    d_dir, d_ckpt = FINAL_DEFAULTS[args.final_input]
    final_dir = args.final_clf_dir or d_dir
    final_ckpt = args.final_clf_ckpt or d_ckpt
    out_dir = args.output_dir or f"outputs/predict_explain_predict_{args.final_input}"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | input tahap 3: {args.final_input} ({final_dir})")
    # Cek checkpoint & kelas model tahap 3 di awal, supaya tidak gagal setelah generasi yang lama
    find_checkpoint(final_dir, final_ckpt)
    {"expl": lambda: import_attr("XLMRobertaExplClassifier", ["train_expl_to_label_xlmr"]),
     "dual": lambda: import_attr("XLMRobertaDualClassifier",
                                 ["train_ph_e_dual_xlmr", "train_phe_to_label_xlmr"]),
     "phe": lambda: (import_attr("XLMRobertaPHEClassifier", ["train_phe_to_label_xlmr"]),
                     import_attr("make_triplet_tokenizer", ["train_phe_to_label_xlmr"])),
     }[args.final_input]()

    # --- Data ---
    raw = pd.read_json(args.test_file, lines=True)
    cols = [c for c in args.explanation_cols if c in raw.columns]
    if not cols:
        raise KeyError(f"Tidak ada kolom penjelasan {args.explanation_cols} di {args.test_file}")
    df = clean_split(raw, args.premise_col, args.hypothesis_col, cols, args.label_col, "test")
    if args.max_samples:
        df = df.head(args.max_samples).reset_index(drop=True)
    premises, hyps, refs = df["premise"].tolist(), df["hypothesis"].tolist(), df["refs"].tolist()
    label2id = {l: i for i, l in enumerate(LABEL_NAMES)}
    y_gold = df["label_str"].map(label2id).values
    results = {"config": {**vars(args), "final_clf_dir": final_dir, "output_dir": out_dir},
               "n": len(df)}

    # --- Tahap 1: PREDICT ---
    clf, clf_tok, clf_len = load_xlmr_classifier(
        XLMRobertaPHClassifier, args.clf_dir, args.clf_ckpt, args.base_model, device)
    y1, p1 = classify(clf, clf_tok, premises, hyps, clf_len, args.clf_batch_size, device)
    del clf
    free_gpu()
    labels1 = [ID2LABEL[i] for i in y1]
    results["stage1"] = label_report(y_gold, y1, "TAHAP 1 — Label awal (P+H) vs label emas")

    # --- Tahap 2: EXPLAIN ---
    gen, gen_tok = load_gemma(args.gen_dir, device)
    print(f"\nGenerasi penjelasan dengan label awal ({len(df)} contoh)...")
    expls = gemma_generate(gen, gen_tok, premises, hyps, labels1,
                           args.gen_batch_size, args.max_new_tokens, args.num_beams)
    del gen
    free_gpu()
    print("\n" + "=" * 60 + "\n   TAHAP 2 — Kualitas penjelasan\n" + "=" * 60)
    results["explanation"] = subset_text_metrics(expls, refs, [True] * len(df), "semua")

    # --- Tahap 3: PREDICT AGAIN (label akhir) ---
    y3, p3 = run_final_stage(args.final_input, final_dir, final_ckpt, args.base_model,
                             premises, hyps, expls, args.clf_batch_size, device)
    results["stage3_final"] = label_report(
        y_gold, y3, f"TAHAP 3 — LABEL AKHIR ({args.final_input}) vs label emas")

    # --- Analisis perubahan label ---
    c1, c3 = (y1 == y_gold), (y3 == y_gold)
    changed = (y1 != y3)
    flips = {
        "berubah": int(changed.sum()),
        "benar_ke_salah": int((c1 & ~c3).sum()),
        "salah_ke_benar": int((~c1 & c3).sum()),
        "salah_ke_salah_lain": int((~c1 & ~c3 & changed).sum()),
    }
    agree = ~changed
    selective = {
        "coverage": float(agree.mean()),
        "accuracy_on_agreed": float(c3[agree].mean()) if agree.any() else None,
        "accuracy_on_disagreed_stage1": float(c1[~agree].mean()) if (~agree).any() else None,
        "accuracy_on_disagreed_stage3": float(c3[~agree].mean()) if (~agree).any() else None,
    }
    results["label_changes"] = flips
    results["selective_prediction"] = selective

    print("\n" + "=" * 60 + "\n   RINGKASAN\n" + "=" * 60)
    print(f"Akurasi tahap 1 (awal)  : {results['stage1']['accuracy']:.4f}")
    print(f"Akurasi tahap 3 (akhir) : {results['stage3_final']['accuracy']:.4f}")
    print(f"Label berubah           : {flips['berubah']} dari {len(df)} "
          f"({flips['berubah'] / len(df):.2%})")
    print(f"  benar -> salah        : {flips['benar_ke_salah']}")
    print(f"  salah -> benar        : {flips['salah_ke_benar']}")
    print(f"  salah -> salah lain   : {flips['salah_ke_salah_lain']}")
    print("Prediksi selektif (hanya jika tahap 1 == tahap 3):")
    print(f"  coverage              : {selective['coverage']:.2%}")
    if selective["accuracy_on_agreed"] is not None:
        print(f"  akurasi (sepakat)     : {selective['accuracy_on_agreed']:.4f}")
    if selective["accuracy_on_disagreed_stage1"] is not None:
        print(f"  akurasi tidak sepakat : tahap1={selective['accuracy_on_disagreed_stage1']:.4f}"
              f" | tahap3={selective['accuracy_on_disagreed_stage3']:.4f}")

    # --- Simpan ---
    out = pd.DataFrame({
        "kalimat1": df["premise"], "kalimat2": df["hypothesis"], "label_emas": df["label_str"],
        "label_tahap1": labels1, "penjelasan_generasi": expls,
        "label_akhir": [ID2LABEL[i] for i in y3],
        "berubah": changed, "tahap1_benar": c1, "akhir_benar": c3,
    })
    for i, lab in ID2LABEL.items():
        out[f"prob1_{lab}"] = p1[:, i]
    for i, lab in ID2LABEL.items():
        out[f"prob3_{lab}"] = p3[:, i]
    for k, c in enumerate(cols):
        out[f"ref_{c}"] = df["refs"].apply(lambda r, k=k: r[k] if k < len(r) else "")
    csv_path = os.path.join(out_dir, "predict_explain_predict_test.csv")
    out.to_csv(csv_path, index=False)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSelesai. CSV: {csv_path}\nMetrik: {out_dir}/results.json")


if __name__ == "__main__":
    main()
