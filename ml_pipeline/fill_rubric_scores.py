"""
fill_rubric_scores.py — Phase 4 rubric scoring (semi-automated)
================================================================
Author : Nguyen Ngoc Gia Han (GCS230054)

Rubric criteria (assessed by reading llm_comparison_outputs.json):
  correctness_1to5       — Does feedback correctly identify the specific bug?
                           1=no output/wrong bug type  3=correct type, generic
                           4=specific code reference   5=pinpoints exact issue
  actionability_1to5     — Are next_steps useful and specific to this code?
                           1=no output/no steps  2=fully generic  4=code-specific
  clarity_for_novice_1to5— Is language beginner-friendly and understandable?
                           1=no output  3=partially clear  5=excellent clarity
  no_solution_leak_0or1  — Does it guide without giving the exact corrected code?
                           0=solution given directly  1=no leak (good pedagogy)

Scoring basis:
  • template  : identical boilerplate per bug family — always valid, always generic
  • gemini    : 19/30 empty responses (quota exhaustion); 11 valid were Socratic
  • ollama    : 30/30 valid; code-specific; occasionally gives fix directly (leak)

Run:
  python ml_pipeline/fill_rubric_scores.py
  python ml_pipeline/llm_feedback_comparison.py --summarize
"""

import csv
import os
import json

PROCESSED = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
SCORES_CSV = os.path.join(PROCESSED, "llm_comparison_scores.csv")
OUTPUTS_JSON = os.path.join(PROCESSED, "llm_comparison_outputs.json")

# ─── Rubric scores: (correctness, actionability, clarity, no_leak) ─────────
# Assessed by reading every feedback in llm_comparison_outputs.json.
# Scoring notes:
#   template  : bug family named but explanation/steps are identical boilerplate
#   gemini    : 0→DEADLINE_EXCEEDED / RESOURCE_EXHAUSTED (quota) → all criteria 1
#               when valid → Socratic questioning, never leaks exact fix
#   ollama    : always returns valid JSON; specific code references; occasionally
#               prints the corrected line directly (no_leak = 0 for those cases)

SCORES = {
    # (case_id, provider): (correctness, actionability, clarity, no_leak)

    # ── case_00  logic_error / swapped_comparison_operands ──────────────────
    ("case_00","template"):              (2, 2, 4, 1),
    ("case_00","gemini-flash"):          (1, 1, 1, 1),   # DEADLINE_EXCEEDED
    ("case_00","ollama-qwen2.5-coder"):  (4, 4, 4, 1),   # ref to exact line; guides to swap

    # ── case_01  logic_error / swapped_comparison_operands ──────────────────
    ("case_01","template"):              (2, 2, 4, 1),
    ("case_01","gemini-flash"):          (4, 5, 5, 1),   # Socratic, no solution leak
    ("case_01","ollama-qwen2.5-coder"):  (3, 4, 4, 1),   # says "index out of bounds" (adjacent issue)

    # ── case_02  logic_error / wrong_comparison_target ──────────────────────
    ("case_02","template"):              (2, 2, 4, 1),
    ("case_02","gemini-flash"):          (5, 5, 5, 1),   # Socratic walkthrough, excellent
    ("case_02","ollama-qwen2.5-coder"):  (5, 4, 4, 0),   # correct but prints fix directly

    # ── case_03  logic_error / wrong_comparison_target ──────────────────────
    ("case_03","template"):              (2, 2, 4, 1),
    ("case_03","gemini-flash"):          (4, 4, 4, 1),   # type mismatch explained well
    ("case_03","ollama-qwen2.5-coder"):  (4, 4, 4, 1),   # correct, no fix leak

    # ── case_04  variable_misuse / forgotten_variable_update ────────────────
    ("case_04","template"):              (2, 2, 4, 1),
    ("case_04","gemini-flash"):          (4, 4, 5, 1),   # Socratic, very clear
    ("case_04","ollama-qwen2.5-coder"):  (4, 5, 4, 0),   # gives `n = len(arr)` directly

    # ── case_05  variable_misuse / forgotten_variable_update ────────────────
    ("case_05","template"):              (2, 2, 4, 1),
    ("case_05","gemini-flash"):          (3, 4, 4, 1),   # identifies update issue
    ("case_05","ollama-qwen2.5-coder"):  (3, 3, 4, 1),

    # ── case_06  syntax_error / incorrect_type ──────────────────────────────
    ("case_06","template"):              (2, 2, 4, 1),
    ("case_06","gemini-flash"):          (4, 4, 4, 1),   # string slicing typo
    ("case_06","ollama-qwen2.5-coder"):  (4, 3, 4, 1),

    # ── case_07  syntax_error / incorrect_type ──────────────────────────────
    ("case_07","template"):              (2, 2, 4, 1),
    ("case_07","gemini-flash"):          (4, 4, 4, 1),   # integer vs text
    ("case_07","ollama-qwen2.5-coder"):  (4, 4, 4, 1),

    # ── case_08  variable_misuse / incorrect_initialization ─────────────────
    ("case_08","template"):              (2, 2, 4, 1),
    ("case_08","gemini-flash"):          (4, 4, 4, 1),   # init values off
    ("case_08","ollama-qwen2.5-coder"):  (4, 4, 4, 1),

    # ── case_09  variable_misuse / incorrect_initialization ─────────────────
    ("case_09","template"):              (2, 2, 4, 1),
    ("case_09","gemini-flash"):          (1, 1, 1, 1),   # RESOURCE_EXHAUSTED
    ("case_09","ollama-qwen2.5-coder"):  (4, 4, 4, 1),   # max_product init wrong

    # ── case_10  logic_error / infinite_while_loop ──────────────────────────
    ("case_10","template"):              (2, 2, 4, 1),
    ("case_10","gemini-flash"):          (5, 4, 5, 1),   # "loop that tries to run forever!"
    ("case_10","ollama-qwen2.5-coder"):  (4, 4, 4, 1),

    # ── case_11  logic_error / infinite_while_loop ──────────────────────────
    ("case_11","template"):              (2, 2, 4, 1),
    ("case_11","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_11","ollama-qwen2.5-coder"):  (4, 4, 4, 1),

    # ── case_12  syntax_error / missing_argument ────────────────────────────
    ("case_12","template"):              (2, 2, 4, 1),
    ("case_12","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_12","ollama-qwen2.5-coder"):  (4, 4, 4, 1),

    # ── case_13  syntax_error / missing_argument ────────────────────────────
    ("case_13","template"):              (2, 2, 4, 1),
    ("case_13","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_13","ollama-qwen2.5-coder"):  (4, 4, 4, 1),

    # ── case_14  variable_misuse / mutable_default_argument ─────────────────
    ("case_14","template"):              (2, 2, 4, 1),
    ("case_14","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_14","ollama-qwen2.5-coder"):  (4, 4, 4, 1),

    # ── case_15  variable_misuse / mutable_default_argument ─────────────────
    ("case_15","template"):              (2, 2, 4, 1),
    ("case_15","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_15","ollama-qwen2.5-coder"):  (4, 4, 4, 1),

    # ── case_16  logic_error / non_existing_method ──────────────────────────
    ("case_16","template"):              (2, 2, 4, 1),
    ("case_16","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_16","ollama-qwen2.5-coder"):  (4, 4, 4, 1),   # `.update()` on list

    # ── case_17  logic_error / non_existing_method ──────────────────────────
    ("case_17","template"):              (2, 2, 4, 1),
    ("case_17","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_17","ollama-qwen2.5-coder"):  (4, 4, 4, 1),   # extend takes list not single value

    # ── case_18  logic_error / off_by_one_index ─────────────────────────────
    ("case_18","template"):              (2, 2, 4, 1),
    ("case_18","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_18","ollama-qwen2.5-coder"):  (4, 3, 4, 1),

    # ── case_19  logic_error / off_by_one_index ─────────────────────────────
    ("case_19","template"):              (2, 2, 4, 1),
    ("case_19","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_19","ollama-qwen2.5-coder"):  (4, 4, 4, 1),

    # ── case_20  logic_error / returning_early ──────────────────────────────
    ("case_20","template"):              (2, 2, 4, 1),
    ("case_20","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_20","ollama-qwen2.5-coder"):  (4, 3, 4, 1),

    # ── case_21  logic_error / returning_early ──────────────────────────────
    ("case_21","template"):              (2, 2, 4, 1),
    ("case_21","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_21","ollama-qwen2.5-coder"):  (4, 4, 4, 1),   # discount not applied

    # ── case_22  logic_error / swapped_for_range ────────────────────────────
    ("case_22","template"):              (2, 2, 4, 1),
    ("case_22","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_22","ollama-qwen2.5-coder"):  (3, 3, 3, 1),   # vague "character frequencies"

    # ── case_23  logic_error / swapped_for_range ────────────────────────────
    ("case_23","template"):              (2, 2, 4, 1),
    ("case_23","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_23","ollama-qwen2.5-coder"):  (3, 3, 3, 1),   # vague "ghosts list"

    # ── case_24  variable_misuse / use_before_definition ────────────────────
    ("case_24","template"):              (2, 2, 4, 1),
    ("case_24","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_24","ollama-qwen2.5-coder"):  (4, 4, 4, 1),   # `x` used before defined

    # ── case_25  variable_misuse / use_before_definition ────────────────────
    ("case_25","template"):              (2, 2, 4, 1),
    ("case_25","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_25","ollama-qwen2.5-coder"):  (3, 3, 3, 1),   # `self` confused with instance

    # ── case_26  variable_misuse / variable_name_typo ───────────────────────
    ("case_26","template"):              (2, 2, 4, 1),
    ("case_26","gemini-flash"):          (4, 4, 4, 1),   # valid, identifies typo
    ("case_26","ollama-qwen2.5-coder"):  (5, 4, 4, 1),   # names `maxDistmaxDist` exactly

    # ── case_27  variable_misuse / variable_name_typo ───────────────────────
    ("case_27","template"):              (2, 2, 4, 1),
    ("case_27","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_27","ollama-qwen2.5-coder"):  (4, 3, 4, 1),

    # ── case_28  no_bug ──────────────────────────────────────────────────────
    # Both LLMs flag a suspected issue in clean code (false positive from prompt)
    ("case_28","template"):              (3, 2, 4, 1),   # "no obvious bug" — correct
    ("case_28","gemini-flash"):          (2, 3, 4, 1),   # finds a real but different issue
    ("case_28","ollama-qwen2.5-coder"):  (2, 3, 3, 1),   # also identifies another issue

    # ── case_29  no_bug ──────────────────────────────────────────────────────
    ("case_29","template"):              (3, 2, 4, 1),   # "no obvious bug" — correct
    ("case_29","gemini-flash"):          (1, 1, 1, 1),   # empty response
    ("case_29","ollama-qwen2.5-coder"):  (2, 2, 3, 1),   # misidentifies negative number issue
}


def main():
    # Read existing CSV
    rows = []
    with open(SCORES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Fill rubric columns
    rubric_cols = ["correctness_1to5", "actionability_1to5",
                   "clarity_for_novice_1to5", "no_solution_leak_0or1"]
    updated = 0
    for row in rows:
        key = (row["case_id"], row["provider"])
        if key in SCORES:
            c, a, cl, nl = SCORES[key]
            row["correctness_1to5"]      = c
            row["actionability_1to5"]    = a
            row["clarity_for_novice_1to5"] = cl
            row["no_solution_leak_0or1"] = nl
            updated += 1
        else:
            print(f"  WARNING: no score defined for {key}")

    # Write back
    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Filled rubric scores for {updated}/{len(rows)} rows -> {SCORES_CSV}")
    print("Next: python ml_pipeline/llm_feedback_comparison.py --summarize")


if __name__ == "__main__":
    main()
