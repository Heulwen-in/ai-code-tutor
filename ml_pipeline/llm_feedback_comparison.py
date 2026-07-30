"""
Phase 4 — Step 8: LLM Feedback Comparison Experiment
====================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  The tutor requirement is to generate feedback with PRE-TRAINED LLMs and to
  compare MORE THAN ONE model. This script runs a fixed set of test cases
  through every available provider (plus the deterministic template baseline)
  with the IDENTICAL prompt, records automatic metrics (latency, JSON
  validity, length, estimated cost), and produces a CSV with empty rubric
  columns for manual scoring (correctness / actionability / clarity /
  no-solution-leak) — the aggregated table goes into the thesis.

Test cases: 2 per fine-grained subtype (28) + 2 clean snippets = 30 cases,
deterministic (seed 42), code < 1500 chars so outputs stay readable.

HOW TO RUN:
  1. Set API keys in backend/.env (ANTHROPIC_API_KEY / GEMINI_API_KEY) and/or
     start Ollama. Providers without credentials are skipped and reported.
  2. python ml_pipeline/llm_feedback_comparison.py          (run the experiment)
  3. Fill the rubric columns in llm_comparison_scores.csv by hand
  4. python ml_pipeline/llm_feedback_comparison.py --summarize

Output (ml_pipeline/data/processed/):
  llm_comparison_cases.json    — the 30 test cases with gold labels
  llm_comparison_outputs.json  — raw responses from every provider
  llm_comparison_scores.csv    — auto metrics + manual rubric columns
  llm_comparison_summary.txt   — aggregated tables
"""

import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── CONFIG ─────────────────────────────────────────────────────────────
PROCESSED_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
BACKEND_DIR   = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\backend"

CASES_PER_SUBTYPE = 2
N_CLEAN_CASES     = 2
MAX_CODE_CHARS    = 1500
RANDOM_SEED       = 42

# role/skill fixed for the main run — one variable at a time
ROLE, SKILL = "student", "novice"

# $/MTok (input, output). Gemini free tier and local Ollama cost nothing.
PRICING = {
    "gemini-flash":         (0.00, 0.00),
    "ollama-qwen2.5-coder": (0.00, 0.00),
    "template":             (0.00, 0.00),
}

RUBRIC_COLUMNS = ["correctness_1to5", "actionability_1to5",
                  "clarity_for_novice_1to5", "no_solution_leak_0or1"]
# ───────────────────────────────────────────────────────────────────────

# Make backend importable (providers + shared prompt) and load its .env
sys.path.insert(0, BACKEND_DIR)

def load_env(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

load_env(os.path.join(BACKEND_DIR, ".env"))

from app.services.llm_feedback import (  # noqa: E402
    GeminiProvider, OllamaProvider, build_prompt,
)
from app.services.feedback_generator import _template_feedback  # noqa: E402


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def est_tokens(text):
    return len(text) / 4  # rough chars->tokens heuristic


# ════════════════════════════════════════════════════════════════
#  --summarize mode: aggregate the manually scored CSV
# ════════════════════════════════════════════════════════════════
if "--summarize" in sys.argv:
    section("Summarising manual rubric scores")
    scores_path = os.path.join(PROCESSED_DIR, "llm_comparison_scores.csv")
    df = pd.read_csv(scores_path)

    filled = df.dropna(subset=RUBRIC_COLUMNS)
    print(f"Rows with completed rubric: {len(filled)} / {len(df)}")

    agg = filled.groupby("provider").agg(
        n=("case_id", "count"),
        correctness=("correctness_1to5", "mean"),
        actionability=("actionability_1to5", "mean"),
        clarity=("clarity_for_novice_1to5", "mean"),
        no_solution_leak=("no_solution_leak_0or1", "mean"),
        mean_latency_s=("latency_s", "mean"),
        json_valid_rate=("json_valid", "mean"),
        mean_words=("word_count", "mean"),
        total_cost_usd=("est_cost_usd", "sum"),
    ).round(3)
    print("\n" + agg.to_string())

    out_path = os.path.join(PROCESSED_DIR, "llm_comparison_summary.txt")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n\nManual rubric aggregation (mean per provider):\n")
        f.write(agg.to_string() + "\n")
    print(f"\n[appended] {out_path}")
    sys.exit(0)


# ════════════════════════════════════════════════════════════════
#  STEP 1 — Build the test-case set
# ════════════════════════════════════════════════════════════════
section("STEP 1 — Selecting test cases")

df_buggy = pd.read_parquet(os.path.join(PROCESSED_DIR, "test14.parquet"))
df_test4 = pd.read_parquet(os.path.join(PROCESSED_DIR, "test4_v2.parquet"))
df_clean = df_test4[df_test4["label"] == 3]

with open(os.path.join(PROCESSED_DIR, "subtype_mapping.json"), encoding="utf-8") as f:
    MAPPING = json.load(f)
HUMAN_NAMES = MAPPING["human_names"]

rng = np.random.default_rng(RANDOM_SEED)
cases = []

for bug_type_name, group in df_buggy.groupby("bug_type"):
    pool = group[group["code"].str.len() < MAX_CODE_CHARS]
    if len(pool) == 0:
        pool = group  # fall back rather than dropping a subtype entirely
        print(f"  NOTE: no short samples for {bug_type_name}; using full pool")
    take = min(CASES_PER_SUBTYPE, len(pool))
    idx = rng.choice(len(pool), size=take, replace=False)
    for i in idx:
        row = pool.iloc[int(i)]
        cases.append({
            "case_id":      f"case_{len(cases):02d}",
            "code":         row["code"],
            "gold_coarse":  row["error_class"],
            "gold_subtype": HUMAN_NAMES.get(row["bug_type"], row["bug_type"]),
        })

clean_pool = df_clean[df_clean["code"].str.len() < MAX_CODE_CHARS]
idx = rng.choice(len(clean_pool), size=N_CLEAN_CASES, replace=False)
for i in idx:
    row = clean_pool.iloc[int(i)]
    cases.append({
        "case_id":      f"case_{len(cases):02d}",
        "code":         row["code"],
        "gold_coarse":  "no_bug",
        "gold_subtype": None,
    })

print(f"Total cases: {len(cases)} "
      f"({len(cases) - N_CLEAN_CASES} buggy + {N_CLEAN_CASES} clean)")

cases_path = os.path.join(PROCESSED_DIR, "llm_comparison_cases.json")
with open(cases_path, "w", encoding="utf-8") as f:
    json.dump(cases, f, indent=2)
print(f"[saved] {cases_path}")


# ════════════════════════════════════════════════════════════════
#  STEP 2 — Determine available providers
# ════════════════════════════════════════════════════════════════
section("STEP 2 — Checking provider availability")

providers = {"template": None}  # template baseline always runs

if os.getenv("GEMINI_API_KEY"):
    providers["gemini-flash"] = GeminiProvider()
else:
    print("  SKIP gemini — GEMINI_API_KEY not set")

try:
    import requests
    requests.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags",
                 timeout=3).raise_for_status()
    providers["ollama-qwen2.5-coder"] = OllamaProvider()
except Exception:
    print("  SKIP ollama — server not reachable")

print(f"Providers in this run: {list(providers.keys())}")
if len(providers) < 3:
    print("  WARNING: fewer than 2 LLM providers available — the comparison "
          "needs at least 2. Add API keys to backend/.env and re-run.")


# ════════════════════════════════════════════════════════════════
#  STEP 3 — Run every case through every provider
# ════════════════════════════════════════════════════════════════
section("STEP 3 — Generating feedback")

# Throttle + retry so free-tier rate limits (HTTP 429 RESOURCE_EXHAUSTED) and
# transient overload (HTTP 503 UNAVAILABLE) don't count as real failures.
_LAST_CALL: dict = {}
MIN_INTERVAL_S = {"gemini-flash": 13.0}   # Gemini free tier allows ~5 req/min
MAX_RETRIES = 6


def _retry_delay_from_error(msg: str):
    import re
    for pat in (r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)",
                r"retry in (\d+(?:\.\d+)?)s"):
        m = re.search(pat, msg)
        if m:
            return float(m.group(1))
    return None


def _generate_with_retry(pname, provider, prompt):
    """Return (feedback, error, latency_s). Paces calls to respect per-minute
    quota and retries on rate-limit / transient-overload responses."""
    interval = MIN_INTERVAL_S.get(pname, 0.0)
    last_error = None
    for attempt in range(MAX_RETRIES):
        if interval:
            gap = interval - (time.time() - _LAST_CALL.get(pname, 0.0))
            if gap > 0:
                time.sleep(gap)
        _LAST_CALL[pname] = time.time()
        t0 = time.time()
        try:
            return provider.generate(prompt), None, round(time.time() - t0, 2)
        except Exception as exc:
            last_error = str(exc)
            transient = any(k in last_error for k in
                            ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                             "500", "INTERNAL"))
            if transient and attempt < MAX_RETRIES - 1:
                delay = _retry_delay_from_error(last_error) or 0.0
                print(f"    retry {attempt + 1}/{MAX_RETRIES - 1} for {pname} "
                      f"after transient error; waiting {max(delay + 1.0, 5.0):.0f}s")
                time.sleep(max(delay + 1.0, 5.0))
                continue
            return None, last_error, round(time.time() - t0, 2)
    return None, last_error, 0.0


outputs = []
for case in cases:
    # The prompt uses gold labels (not model predictions) so feedback quality
    # is measured independently of classifier errors.
    coarse  = case["gold_coarse"]
    subtype = case["gold_subtype"]
    prompt  = build_prompt(case["code"], coarse, subtype, None, 0.95, ROLE, SKILL)

    for pname, provider in providers.items():
        error = None
        feedback = None
        if pname == "template":
            t0 = time.time()
            try:
                feedback = _template_feedback(coarse, ROLE, SKILL, 0.95, None, subtype)
            except Exception as exc:
                error = str(exc)
            latency = round(time.time() - t0, 2)
        else:
            feedback, error, latency = _generate_with_retry(pname, provider, prompt)

        fb_dict = feedback.model_dump() if feedback is not None else None
        response_text = json.dumps(fb_dict) if fb_dict else ""
        in_price, out_price = PRICING.get(pname, (0.0, 0.0))
        cost = (est_tokens(prompt) * in_price
                + est_tokens(response_text) * out_price) / 1_000_000

        outputs.append({
            "case_id":     case["case_id"],
            "provider":    pname,
            "gold_coarse": coarse,
            "gold_subtype": subtype,
            "feedback":    fb_dict,
            "json_valid":  int(feedback is not None),
            "latency_s":   latency,
            "word_count":  len((fb_dict["explanation"] if fb_dict else "").split()),
            "est_cost_usd": round(cost, 6),
            "error":       error,
        })
        status = "ok" if feedback is not None else f"FAIL ({error or 'invalid JSON'})"
        print(f"  {case['case_id']} x {pname:22s} [{latency:6.2f}s] {status}")

outputs_path = os.path.join(PROCESSED_DIR, "llm_comparison_outputs.json")
with open(outputs_path, "w", encoding="utf-8") as f:
    json.dump(outputs, f, indent=2)
print(f"\n[saved] {outputs_path}")


# ════════════════════════════════════════════════════════════════
#  STEP 4 — Scores CSV (auto metrics + empty manual rubric)
# ════════════════════════════════════════════════════════════════
section("STEP 4 — Writing rubric CSV")

scores_path = os.path.join(PROCESSED_DIR, "llm_comparison_scores.csv")
auto_cols = ["case_id", "provider", "gold_coarse", "gold_subtype",
             "latency_s", "word_count", "json_valid", "est_cost_usd",
             "summary"]
with open(scores_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(auto_cols + RUBRIC_COLUMNS)
    for o in outputs:
        writer.writerow([
            o["case_id"], o["provider"], o["gold_coarse"], o["gold_subtype"] or "",
            o["latency_s"], o["word_count"], o["json_valid"], o["est_cost_usd"],
            (o["feedback"] or {}).get("summary", ""),
        ] + [""] * len(RUBRIC_COLUMNS))
print(f"[saved] {scores_path}")
print("  -> Fill the 4 rubric columns manually, then re-run with --summarize")


# ════════════════════════════════════════════════════════════════
#  STEP 5 — Auto-metric summary
# ════════════════════════════════════════════════════════════════
section("STEP 5 — Automatic metrics summary")

df_out = pd.DataFrame(outputs)
agg = df_out.groupby("provider").agg(
    n=("case_id", "count"),
    json_valid_rate=("json_valid", "mean"),
    mean_latency_s=("latency_s", "mean"),
    mean_words=("word_count", "mean"),
    total_cost_usd=("est_cost_usd", "sum"),
).round(4)
print("\n" + agg.to_string())

summary_path = os.path.join(PROCESSED_DIR, "llm_comparison_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("LLM Feedback Comparison — Automatic Metrics\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Cases: {len(cases)} ({len(cases) - N_CLEAN_CASES} buggy + "
            f"{N_CLEAN_CASES} clean), role={ROLE}, skill={SKILL}\n")
    f.write(f"Providers: {list(providers.keys())}\n\n")
    f.write(agg.to_string() + "\n")
    f.write("\nManual rubric: fill llm_comparison_scores.csv, then run\n")
    f.write("  python ml_pipeline/llm_feedback_comparison.py --summarize\n")
print(f"\n[saved] {summary_path}")

print(f"""
{'=' * 60}
  LLM FEEDBACK COMPARISON RUN COMPLETE
{'=' * 60}
  Next: score the rubric columns in llm_comparison_scores.csv,
  then run this script again with --summarize.
{'=' * 60}
""")
