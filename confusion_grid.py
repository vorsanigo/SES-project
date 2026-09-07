"""
Confusion matrix grid across models and prompt configurations.

The valid (features, prompt) pairs are NOT a full cross-product:
  - only_text  ->  simple, complex, complex_with_direction
  - all        ->  specific_all_features

So each model has exactly 4 runs: the 3 text-only prompts + the 1 feature prompt.
Grid layout: rows = models, columns = the 4 prompt configurations.
"""

import glob
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

MODELS    = ["Llama-8B", "Llama-70B", "GPT-4o-mini", "GPT-4o"]
N_CLASSES = '3'
LABELS    = ["low", "medium", "high"] if N_CLASSES == '3' else ["low", "high"]

# valid (features, prompt, column_label) combinations
CONFIGS = [
    ("only_text", "simple",                 "simple"),
    ("only_text", "complex",                "complex"),
    #("only_text", "complex_with_direction", "complex\n+direction"),
    ("all",       "specific_all_features",  "features"),
]

'''CONFIGS = [
    ("Prompt 1", "Prompt 2",                 "Prompt 3"),
    ("Prompt 1", "Prompt 2",                "Prompt 3"),
    #("only_text", "complex_with_direction", "complex\n+direction"),
    ("Prompt 1",       "Prompt 2",  "Prompt 3"),
]'''

def csv_path(model, features, prompt, n_classes):
    patterns = {
        "Llama-8B":    f"predictions_llama/classes_{n_classes}/*8B*_{features}_prompt_{prompt}_{n_classes}_classes.csv",
        "Llama-70B":   f"predictions_llama/classes_{n_classes}/*70B*_{features}_prompt_{prompt}_{n_classes}_classes.csv",
        "GPT-4o-mini": f"predictions_gpt/classes_{n_classes}/gpt_gpt-4o-mini_zero_predictions_{features}_prompt_{prompt}_{n_classes}_classes.csv",
        "GPT-4o": f"predictions_gpt/classes_{n_classes}/gpt_gpt-4o_zero_predictions_{features}_prompt_{prompt}_{n_classes}_classes.csv"
    }
    hits = glob.glob(patterns[model])
    return hits[0] if hits else None



n_rows, n_cols = len(MODELS), len(CONFIGS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)

summary = []

for r, model in enumerate(MODELS):
    print(model)
    for c, (features, prompt, col_label) in enumerate(CONFIGS):
        ax = axes[r, c]
        path = csv_path(model, features, prompt, N_CLASSES)

        print(features, prompt, col_label)
        print(path)

        if path is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(col_label, fontsize=10)
            if c == 0: ax.set_ylabel(model, fontsize=10, fontweight="bold")
            continue

        df = pl.read_csv(path)
        print(df)
        y_true = df["true"].to_list()
        y_pred = df["pred"].to_list()

        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        print(c, (features, prompt, col_label))
        print(cm)
        print("\n")
        row_tot = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_tot

        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{cm_norm[i,j]:.0%}", #{cm[i,j]}\n
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i,j] > 0.5 else "black")

        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, fontsize=10)
        ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=10)
        if r == n_rows - 1: ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
        if c == 0:          ax.set_ylabel(f"{model}\nTrue", fontsize=12, fontweight="bold")
        if r == 0:          
            if col_label == 'simple':
                ax.set_title('Simple Prompt', fontsize=12, fontweight="bold")
            elif col_label == 'complex':
                ax.set_title('Intermediate Prompt', fontsize=12, fontweight="bold")
            elif col_label == 'features':
                ax.set_title('Complex Prompt', fontsize=12, fontweight="bold")

        hi, lo = LABELS.index("high"), LABELS.index("low")
        high_to_low = cm[hi, lo] / max(cm[hi].sum(), 1)

        summary.append({
            "model":  model,
            "config": col_label.replace("\n", " "),
            "features": features,
            "prompt": prompt,
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "high_to_low_rate": round(high_to_low, 4),
            "n": len(y_true),
        })

#fig.suptitle(f"Confusion matrices ({N_CLASSES}-class): rows = model, columns = prompt config\n"
#             "cells show count and row-normalised recall", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out_png = f"llm_confusion_matrices/confusion_grid_{N_CLASSES}classes_NO_COUNT.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out_png)

summary_df = pl.DataFrame(summary)
print("\n=== full summary ===")
print(summary_df.sort(["model", "config"]))
print("\n=== high->low error rate (rows=model, cols=config) ===")
print(summary_df.pivot(values="high_to_low_rate", index="model", on="config"))
print("\n=== F1 macro (rows=model, cols=config) ===")
print(summary_df.pivot(values="f1_macro", index="model", on="config"))
summary_df.write_csv(f"llm_confusion_matrices/confusion_summary_{N_CLASSES}classes.csv")
print(f"\nsaved confusion_summary_{N_CLASSES}classes.csv")
