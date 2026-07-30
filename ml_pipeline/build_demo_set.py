"""
Phase 5 — Demo Set Builder
==========================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Builds a varied, presentation-ready set of demo snippets drawn from the UNSEEN
test split (test14_v3 / test4_v3), runs every snippet through the live backend
classifier (Stage 1 -> Stage 2 -> line detection), and writes a Markdown demo
sheet plus a CSV. Covers all 14 fine-grained subtypes (up to PER_SUBTYPE each)
and no_bug, so a viva demonstration exercises the whole pipeline.

HOW TO RUN (from backend/, so .env + app import resolve):
  cd Tutor_AI_code/backend
  ../.venv/Scripts/python.exe ../ml_pipeline/build_demo_set.py

Output (ml_pipeline/data/processed/):
  demo_set.md   — paste-ready snippets, grouped by subtype, expected vs system
  demo_set.csv  — one row per snippet with the prediction outcome
"""

import os
import sys
import json
import random

import pandas as pd

PROCESSED_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
PER_SUBTYPE   = int(os.getenv("PER_SUBTYPE", "5"))   # bug snippets per subtype
N_NOBUG       = int(os.getenv("N_NOBUG", "6"))
MAX_LEN       = int(os.getenv("MAX_LEN", "500"))     # keep snippets readable on a slide
SEED          = 42

# Import the live backend classifier (run from backend/ so .env is loaded there).
sys.path.insert(0, os.getcwd())
from app.services import bug_classifier as bc  # noqa: E402


def main():
    random.seed(SEED)
    bc.load_model()

    with open(os.path.join(PROCESSED_DIR, "subtype_mapping_v3.json"), encoding="utf-8") as f:
        mapping = json.load(f)
    int_to_sub = {int(k): v for k, v in mapping["int_to_subtype"].items()}
    human = mapping["human_names"]

    df14 = pd.read_parquet(os.path.join(PROCESSED_DIR, "test14_v3.parquet"))
    df4  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test4_v3.parquet"))
    gold = pd.read_parquet(os.path.join(PROCESSED_DIR, "line_test_v3.parquet"))[["code", "buggy_lines"]]
    gold_map = {r["code"]: [int(x) for x in r["buggy_lines"]] for _, r in gold.iterrows()}

    rows = []

    # ── 14 bug subtypes (grouped by coarse class order) ──
    for sub_id in range(14):
        transformer = int_to_sub[sub_id]
        pool = df14[(df14["sub_label"] == sub_id) & (df14["code_len"] <= MAX_LEN)]
        pool = pool.sort_values("code_len")
        take = pool.head(PER_SUBTYPE * 3).sample(min(PER_SUBTYPE, len(pool)), random_state=SEED) \
            if len(pool) > PER_SUBTYPE else pool
        for _, r in take.iterrows():
            res = bc.classify(r["code"])
            rows.append({
                "group": human.get(transformer, transformer),
                "coarse_expected": r["error_class"],
                "subtype_expected": human.get(transformer, transformer),
                "code": r["code"],
                "gold_lines": gold_map.get(r["code"]),
                "pred_coarse": res.bug_type,
                "pred_subtype": res.bug_subtype,
                "pred_line": res.line_number,
                "confidence": res.confidence,
                "coarse_ok": res.bug_type == r["error_class"],
                "subtype_ok": res.bug_subtype == human.get(transformer, transformer),
            })

    # ── no_bug (clean code) ──
    nb = df4[(df4["label"] == 3) & (df4["code_len"] <= MAX_LEN)].sample(
        min(N_NOBUG, int((df4["label"] == 3).sum())), random_state=SEED)
    for _, r in nb.iterrows():
        res = bc.classify(r["code"])
        rows.append({
            "group": "no_bug", "coarse_expected": "no_bug", "subtype_expected": "-",
            "code": r["code"], "gold_lines": None,
            "pred_coarse": res.bug_type, "pred_subtype": res.bug_subtype,
            "pred_line": res.line_number, "confidence": res.confidence,
            "coarse_ok": res.bug_type == "no_bug", "subtype_ok": None,
        })

    out = pd.DataFrame(rows)

    # ── Summary ──
    bug = out[out["coarse_expected"] != "no_bug"]
    nbo = out[out["coarse_expected"] == "no_bug"]
    print("\n" + "=" * 64)
    print(f"  DEMO SET — {len(out)} snippets (unseen test problems)")
    print("=" * 64)
    print(f"  Coarse-class correct : {int(out['coarse_ok'].sum())}/{len(out)} "
          f"({out['coarse_ok'].mean()*100:.1f}%)")
    print(f"  Subtype correct (bugs): {int(bug['subtype_ok'].sum())}/{len(bug)} "
          f"({bug['subtype_ok'].mean()*100:.1f}%)")
    print(f"  no_bug correct        : {int(nbo['coarse_ok'].sum())}/{len(nbo)}")
    print("\n  Per-subtype coarse accuracy:")
    for g, sub in bug.groupby("group"):
        print(f"    {g:30s}: {int(sub['coarse_ok'].sum())}/{len(sub)} "
              f"subtype {int(sub['subtype_ok'].sum())}/{len(sub)}")

    # ── CSV ──
    csv_cols = ["group", "coarse_expected", "subtype_expected", "pred_coarse",
                "pred_subtype", "pred_line", "gold_lines", "confidence",
                "coarse_ok", "subtype_ok"]
    out[csv_cols].to_csv(os.path.join(PROCESSED_DIR, "demo_set.csv"), index=False)

    # ── Markdown ──
    md = ["# Demo Set — AI Programming Tutor (v3 models)", "",
          "Snippets drawn from the **unseen** grouped test split (`test14_v3` / `test4_v3`).",
          "Paste each into the analyzer. PASS = system matched the expected label.", "",
          f"**Overall:** coarse {int(out['coarse_ok'].sum())}/{len(out)}, "
          f"subtype (bugs) {int(bug['subtype_ok'].sum())}/{len(bug)}.", ""]
    n = 0
    for g in list(dict.fromkeys(out["group"])):   # preserve subtype order
        sub = out[out["group"] == g]
        md.append(f"## {g}  ({int(sub['coarse_ok'].sum())}/{len(sub)} coarse-correct)\n")
        for _, r in sub.iterrows():
            n += 1
            cok = "PASS" if r["coarse_ok"] else "MISS"
            sline = f" · line {int(r['pred_line'])}" if pd.notna(r["pred_line"]) else ""
            ssub = f" · subtype `{r['pred_subtype']}`" if isinstance(r["pred_subtype"], str) else ""
            gl = f" · gold line(s) {list(r['gold_lines'])}" if isinstance(r["gold_lines"], list) else ""
            md.append(f"**#{n} — expected `{r['coarse_expected']}`"
                      f"{'/`'+r['subtype_expected']+'`' if r['subtype_expected'] != '-' else ''}"
                      f"{gl}**  ")
            md.append(f"System: {cok} `{r['pred_coarse']}` (conf {r['confidence']:.3f}){ssub}{sline}\n")
            md.append("```python")
            md.append(str(r["code"]).rstrip())
            md.append("```\n")
    with open(os.path.join(PROCESSED_DIR, "demo_set.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n[saved] {os.path.join(PROCESSED_DIR, 'demo_set.md')}")
    print(f"[saved] {os.path.join(PROCESSED_DIR, 'demo_set.csv')}")


if __name__ == "__main__":
    main()
