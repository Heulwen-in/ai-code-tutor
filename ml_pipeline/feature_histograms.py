"""
Plots a histogram for each of the 40 extracted features,
coloured by error class (syntax / logic / variable misuse).
Shows distribution shape, overlap, and class separability.

HOW TO RUN:
  python ml_pipeline/feature_histograms.py

Output:
  ml_pipeline/data/processed/feature_histograms.png
  ml_pipeline/data/processed/feature_histograms_behaviour.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────
PROCESSED_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
OUTPUT_DIR    = PROCESSED_DIR
# ────────────────────────────────────────────────────────────────

print("Loading training features...")
df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_features.parquet"))

CLASSES    = ["syntax_error", "logic_error", "variable_misuse"]
COLORS     = {"syntax_error": "#4C72B0", "logic_error": "#55A868", "variable_misuse": "#C44E52"}
LABELS     = {"syntax_error": "Syntax", "logic_error": "Logic", "variable_misuse": "Variable misuse"}

feature_cols = [c for c in df.columns if c not in ["label", "error_class"]]
print(f"Total features: {len(feature_cols)}")

# Split by class
dfs = {cls: df[df["error_class"] == cls] for cls in CLASSES}

# ── Helper: clip to 99th percentile for readability ─────────────
def clip_p99(series):
    return series.clip(upper=series.quantile(0.99))

# ════════════════════════════════════════════════════════════════
#  CHART 1 — All 40 features in a grid (8 cols × 5 rows)
# ════════════════════════════════════════════════════════════════
print("\nGenerating full feature histogram grid (40 features)...")

NCOLS = 8
NROWS = int(np.ceil(len(feature_cols) / NCOLS))
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(28, NROWS * 3.2))
axes = axes.flatten()

for i, feat in enumerate(feature_cols):
    ax = axes[i]
    for cls in CLASSES:
        data = clip_p99(dfs[cls][feat].dropna())
        bins = min(30, data.nunique())
        if bins < 2:
            bins = 2
        ax.hist(data, bins=bins, alpha=0.55,
                color=COLORS[cls], label=LABELS[cls], density=True)

    ax.set_title(feat, fontsize=8, fontweight="bold", pad=3)
    ax.set_xlabel("Value", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

# Hide unused axes
for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)

# Legend
legend_patches = [mpatches.Patch(color=COLORS[c], label=LABELS[c], alpha=0.7)
                  for c in CLASSES]
fig.legend(handles=legend_patches, loc="lower right",
           fontsize=10, framealpha=0.9, ncol=3)

fig.suptitle("Feature Distributions by Error Class — All 40 Features\n"
             "(density normalised; clipped at 99th percentile)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()

out1 = os.path.join(OUTPUT_DIR, "feature_histograms.png")
plt.savefig(out1, dpi=130, bbox_inches="tight")
plt.close()
print(f"[saved] {out1}")


# ════════════════════════════════════════════════════════════════
#  CHART 2 — Behaviour features only (focused view)
# ════════════════════════════════════════════════════════════════
print("\nGenerating behaviour feature histograms...")

behaviour_features = [
    "max_indent_depth", "mean_indent_depth", "single_letter_vars",
    "has_type_hints", "has_list_comp", "has_generator",
    "has_recursion", "magic_numbers", "builtins_used",
    "descriptive_ratio", "comment_ratio"
]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, feat in enumerate(behaviour_features):
    ax = axes[i]
    for cls in CLASSES:
        data = clip_p99(dfs[cls][feat].dropna())
        bins = min(25, max(data.nunique(), 2))
        ax.hist(data, bins=bins, alpha=0.6,
                color=COLORS[cls], label=LABELS[cls], density=True)

    ax.set_title(feat.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.set_xlabel("Value", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(labelsize=8)

for j in range(len(behaviour_features), len(axes)):
    axes[j].set_visible(False)

legend_patches = [mpatches.Patch(color=COLORS[c], label=LABELS[c], alpha=0.7)
                  for c in CLASSES]
fig.legend(handles=legend_patches, loc="lower right",
           fontsize=11, framealpha=0.9)

fig.suptitle("Behaviour Feature Distributions by Error Class\n"
             "(signals used by skill_detector.py for novice/professional classification)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

out2 = os.path.join(OUTPUT_DIR, "feature_histograms_behaviour.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {out2}")


# ════════════════════════════════════════════════════════════════
#  CHART 3 — Top 12 most discriminative features (highlighted)
# ════════════════════════════════════════════════════════════════
print("\nGenerating top discriminative features chart...")

# Features with highest between-class variance
means = df.groupby("error_class")[feature_cols].mean()
between_var = means.var(axis=0).sort_values(ascending=False)
top12 = between_var.head(12).index.tolist()

print("Top 12 discriminative features (highest between-class variance):")
for feat in top12:
    print(f"  {feat}: {between_var[feat]:.4f}")

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, feat in enumerate(top12):
    ax = axes[i]
    for cls in CLASSES:
        data = clip_p99(dfs[cls][feat].dropna())
        bins = min(30, max(data.nunique(), 2))
        ax.hist(data, bins=bins, alpha=0.6,
                color=COLORS[cls], label=LABELS[cls], density=True)

    # Add mean lines
    for cls in CLASSES:
        mn = dfs[cls][feat].mean()
        ax.axvline(mn, color=COLORS[cls], linestyle="--", linewidth=1.2, alpha=0.8)

    between = between_var[feat]
    ax.set_title(f"{feat}\n(between-class var={between:.4f})",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)

legend_patches = [mpatches.Patch(color=COLORS[c], label=LABELS[c], alpha=0.7)
                  for c in CLASSES]
fig.legend(handles=legend_patches, loc="lower right",
           fontsize=11, framealpha=0.9)
fig.text(0.5, 1.0, "Dashed lines = class mean", ha="center",
         fontsize=10, color="gray")

fig.suptitle("Top 12 Most Discriminative Features\n"
             "(sorted by between-class variance; dashed = class mean)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

out3 = os.path.join(OUTPUT_DIR, "feature_histograms_top12.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {out3}")

print(f"""
{'=' * 60}
  FEATURE HISTOGRAMS COMPLETE
{'=' * 60}
  Saved 3 charts:
    feature_histograms.png         — all 40 features grid
    feature_histograms_behaviour.png — behaviour features only
    feature_histograms_top12.png   — top 12 discriminative

  Key findings to note in report:
  - Features with wide class separation → useful for classifier
  - Features with overlapping distributions → classifier struggles
  - Behaviour features → feed into skill_detector.py
{'=' * 60}
""")