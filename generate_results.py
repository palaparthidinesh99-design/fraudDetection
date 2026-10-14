#!/usr/bin/env python3
"""
generate_results.py  — Parallelised Replication Pipeline
---------------------------------------------------------
Runs all 150 experiments in parallel using all M5 CPU cores (joblib).
Each run: resample → train → threshold-calibrate → evaluate.
Results are streamed to CSV as they complete.

Usage:
    python generate_results.py          # full parallel run
    python generate_results.py --resume # skip already-done runs
    python generate_results.py --jobs 4 # override core count
"""

import argparse
import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import prepare_train_test_split
from src.evaluator import evaluate_single_combination
from src.config import (
    MODEL_CONFIGS, OVERSAMPLER_NAMES, SEEDS,
    RESULTS_DIR, PLOTS_DIR, PAPER_TABLE2_BENCHMARKS
)
from src.samplers import get_sampler
from src.models import get_model

MODEL_ORDER = ["Logistic Regression", "LightGBM", "XGBoost", "MLP", "Random Forest"]

SAMPLER_COLORS = {
    "SMOTE":                           "#1f77b4",
    "Cost-sensitive SMOTE":            "#aec7e8",
    "ADASYN":                          "#ff7f0e",
    "Cost-sensitive ADASYN":           "#ffbb78",
    "Borderline-SMOTE":                "#2ca02c",
    "Cost-sensitive Borderline-SMOTE": "#98df8a",
    "SMOTE-ENN":                       "#d62728",
    "Cost-sensitive SMOTE-ENN":        "#ff9896",
    "SMOTE-Tomek":                     "#9467bd",
    "Cost-sensitive SMOTE-Tomek":      "#c5b0d5",
}

METRIC_COLS = ["Accuracy", "Specificity", "Sensitivity", "F1", "TCS", "ACM", "AUC"]


# ─── Parallel worker ────────────────────────────────────────────────────────

def _run_one(model_name, sampler_name, seed, data):
    """Single experiment worker — standard evaluation without threshold gaming."""
    try:
        res = evaluate_single_combination(
            model_name=model_name,
            sampler_name=sampler_name,
            data=data,
            random_state=seed,
            threshold=None,
            parallel_mode=False,
        )
        res["_status"] = "ok"
    except Exception as e:
        res = {
            "Model": model_name, "Sampler": sampler_name,
            "RandomState": seed, "_status": f"ERROR: {e}",
        }
    return res


# ─── Plotting helpers ────────────────────────────────────────────────────────

def _plot_figure2(df_mean):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODEL_ORDER:
        df_m = df_mean[df_mean["Model"] == model].copy()
        if df_m.empty:
            continue
        df_m["Sampler"] = pd.Categorical(df_m["Sampler"], categories=OVERSAMPLER_NAMES, ordered=True)
        df_m = df_m.sort_values("Sampler").dropna(subset=["Sampler"])

        x = np.arange(len(df_m))
        w = 0.38
        colors = [SAMPLER_COLORS.get(s, "#888") for s in df_m["Sampler"]]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - w/2, df_m["TCS"], w, color=colors, alpha=0.92, edgecolor="white", label="TCS")
        ax.bar(x + w/2, df_m["ACM"], w, color=colors, alpha=0.50, hatch="//", edgecolor="white", label="ACM")

        ax.set_ylabel("Value (Log Scale)", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.set_title(f"({model}) — TCS and ACM per Oversampling Method",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(df_m["Sampler"], rotation=38, ha="right", fontsize=9)
        ax.legend(["TCS (solid)", "ACM (hatched)"], loc="upper right", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        plt.tight_layout()
        fname1 = PLOTS_DIR / f"figure_2_{model.lower().replace(' ', '_')}_tcs_acm.png"
        fname2 = RESULTS_DIR / f"figure_2_{model.lower().replace(' ', '_')}_tcs_acm.png"
        plt.savefig(fname1, dpi=200, bbox_inches="tight")
        plt.savefig(fname2, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname1.name} (in plots/ and results/)")


def _plot_roc(model, data, seed=42):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    for sampler_name in OVERSAMPLER_NAMES:
        from src.evaluator import _get_resampled_dataset
        X_res, y_res = _get_resampled_dataset(
            sampler_name, seed, data["X_train"], data["y_train"], data["amounts_train"]
        )
        clf = get_model(model, random_state=seed)
        clf.fit(X_res, y_res)

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(data["X_test"])[:, 1]
        elif hasattr(clf, "decision_function"):
            df_v = clf.decision_function(data["X_test"])
            y_prob = 1.0 / (1.0 + np.exp(-df_v))
        else:
            y_prob = clf.predict(data["X_test"]).astype(float)

        fpr, tpr, _ = roc_curve(data["y_test"], y_prob)
        roc_auc = sk_auc(fpr, tpr)
        is_cs = "Cost-sensitive" in sampler_name
        ax.plot(fpr, tpr,
                color=SAMPLER_COLORS.get(sampler_name, "#666"),
                linestyle="-" if is_cs else "--",
                lw=2.0 if is_cs else 1.5,
                label=f"{sampler_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlim([0.0, 0.15]); ax.set_ylim([0.70, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fname1 = PLOTS_DIR / f"roc_curve_{model.lower().replace(' ', '_')}.png"
    fname2 = RESULTS_DIR / f"roc_curve_{model.lower().replace(' ', '_')}.png"
    plt.savefig(fname1, dpi=200, bbox_inches="tight")
    plt.savefig(fname2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname1.name} (in plots/ and results/)")


# ─── Table output ────────────────────────────────────────────────────────────

def _save_tables(df_mean, bench):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Markdown table
    lines = ["# Table 2 Replication — Our Results\n"]
    for model in MODEL_ORDER:
        df_m = df_mean[df_mean["Model"] == model].copy()
        df_m["Sampler"] = pd.Categorical(df_m["Sampler"], categories=OVERSAMPLER_NAMES, ordered=True)
        df_m = df_m.sort_values("Sampler")
        lines += [f"\n## {model}\n",
                  "| Sampler | Accuracy | Specificity | Sensitivity | F1 | TCS | ACM | AUC |",
                  "|---------|----------|-------------|-------------|-----|-----|-----|-----|"]
        for _, row in df_m.iterrows():
            lines.append(
                f"| {row['Sampler']} | {row['Accuracy']:.4f} | {row['Specificity']:.4f} | "
                f"{row['Sensitivity']:.4f} | {row['F1']:.4f} | {row['TCS']:.2f} | "
                f"{row['ACM']:.2f} | {row['AUC']:.4f} |"
            )
    (RESULTS_DIR / "table2_our_results.md").write_text("\n".join(lines))
    print(f"  Saved: table2_our_results.md")

    # 2. Side-by-side comparison CSV
    rows = []
    for model in MODEL_ORDER:
        for sampler in OVERSAMPLER_NAMES:
            our = df_mean[(df_mean["Model"] == model) & (df_mean["Sampler"] == sampler)]
            paper = bench.get((model, sampler), {})
            if our.empty:
                continue
            r = our.iloc[0]
            rows.append({
                "Model": model, "Sampler": sampler,
                "Our_Acc":  f"{r['Accuracy']:.4f}",   "Paper_Acc":  paper.get("Accuracy", ""),
                "Our_Spec": f"{r['Specificity']:.4f}", "Paper_Spec": paper.get("Specificity", ""),
                "Our_Sens": f"{r['Sensitivity']:.4f}", "Paper_Sens": paper.get("Sensitivity", ""),
                "Our_F1":   f"{r['F1']:.4f}",          "Paper_F1":   paper.get("F1", ""),
                "Our_TCS":  f"{r['TCS']:.2f}",         "Paper_TCS":  paper.get("TCS", ""),
                "Our_ACM":  f"{r['ACM']:.2f}",         "Paper_ACM":  paper.get("ACM", ""),
                "Our_AUC":  f"{r['AUC']:.4f}",
            })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "table2_comparison.csv", index=False)
    print(f"  Saved: table2_comparison.csv")


def _print_comparison(df_mean, bench):
    try:
        from tabulate import tabulate
    except ImportError:
        return

    header = ["Model", "Sampler", "Sens(Ours)", "Sens(Paper)", "F1(Ours)", "F1(Paper)",
              "TCS(Ours)", "TCS(Paper)", "ACM(Ours)", "ACM(Paper)"]
    rows = []
    for model in MODEL_ORDER:
        for sampler in OVERSAMPLER_NAMES:
            our = df_mean[(df_mean["Model"] == model) & (df_mean["Sampler"] == sampler)]
            paper = bench.get((model, sampler), {})
            if our.empty:
                continue
            r = our.iloc[0]
            rows.append([
                model[:14], sampler[:30],
                f"{r['Sensitivity']:.3f}", f"{paper.get('Sensitivity','?')}",
                f"{r['F1']:.3f}",          f"{paper.get('F1','?')}",
                f"{r['TCS']:.0f}",         f"{paper.get('TCS','?')}",
                f"{r['ACM']:.2f}",         f"{paper.get('ACM','?')}",
            ])

    print("\n" + "="*110)
    print("TABLE 2: OUR RESULTS vs PAPER")
    print("="*110)
    print(tabulate(rows, headers=header, tablefmt="github"))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",    action="store_true", help="Skip completed runs")
    parser.add_argument("--skip-roc",  action="store_true", help="Skip ROC curve generation")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Parallel jobs (-1 = all cores, default)")
    args = parser.parse_args()

    print("=" * 70)
    print(" REPLICATION — PARALLEL PIPELINE  (Apple M5, all cores)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs_path  = RESULTS_DIR / "all_runs_raw.csv"
    mean_path  = RESULTS_DIR / "mean_benchmark_results.csv"

    # Build task list
    bench = PAPER_TABLE2_BENCHMARKS
    all_tasks = [
        (m, s, seed)
        for m in MODEL_ORDER
        for s in OVERSAMPLER_NAMES
        for seed in SEEDS
    ]

    # Handle resume
    existing_keys = set()
    existing_records = []
    if args.resume and runs_path.exists():
        df_ex = pd.read_csv(runs_path)
        for _, row in df_ex.iterrows():
            k = (row["Model"], row["Sampler"], int(row["RandomState"]))
            existing_keys.add(k)
            existing_records.append(row.to_dict())
        print(f"Resuming: {len(existing_keys)}/{len(all_tasks)} runs already cached.")

    todo = [(m, s, seed) for m, s, seed in all_tasks
            if (m, s, int(seed)) not in existing_keys]

    print(f"\n[1/4] Loading dataset...")
    data = prepare_train_test_split()
    print(f"  Train {data['X_train'].shape} | Test {data['X_test'].shape} | "
          f"Fraud train:{data['y_train'].sum()} test:{data['y_test'].sum()}")

    print(f"\n[2/4] Running {len(todo)} experiments...")
    t_start = time.time()

    all_records = list(existing_records)
    df_mean = None

    for model_idx, model in enumerate(MODEL_ORDER, 1):
        model_tasks = [(m, s, seed) for (m, s, seed) in todo if m == model]
        if not model_tasks:
            print(f"\n({model_idx}/{len(MODEL_ORDER)}) {model} — all runs already cached.")
            continue

        print(f"\n({model_idx}/{len(MODEL_ORDER)}) Starting {model} ({len(model_tasks)} runs)...")
        t_model_start = time.time()
        for (m, s, seed) in model_tasks:
            t_run = time.time()
            res = _run_one(m, s, seed, data)
            dur = time.time() - t_run
            if res.get("_status") == "ok":
                all_records.append(res)
                sens_val = res.get("Sensitivity", 0.0)
                f1_val = res.get("F1", 0.0)
                print(f"  {res['Model']:18s} | {res['Sampler'][:25]:<25} | seed={res['RandomState']:<4} | "
                      f"Sens={sens_val:.3f} | F1={f1_val:.3f} | ({dur:.1f}s)")
                # Incremental write to CSV
                df_runs = pd.DataFrame(all_records)
                save_cols = [c for c in df_runs.columns if c != "_status"]
                df_runs[save_cols].to_csv(runs_path, index=False)

                df_mean = df_runs.groupby(["Model", "Sampler"])[METRIC_COLS].mean().reset_index()
                df_mean.to_csv(mean_path, index=False)
                _save_tables(df_mean, bench)
            else:
                print(f"  ERROR: {res.get('_status')}")

        t_model_dur = time.time() - t_model_start
        print(f"  --> Completed {model} in {t_model_dur:.1f}s")

        # Update plots after each model finishes
        if df_mean is not None and not df_mean.empty:
            _plot_figure2(df_mean)
        if not args.skip_roc:
            print(f"  Generating ROC curve for {model}...")
            _plot_roc(model, data, seed=SEEDS[0])

    elapsed = time.time() - t_start
    print(f"\n[3/4] Completed all experiments in {elapsed:.1f}s")

    # Final summary and table
    if all_records:
        df_runs = pd.DataFrame(all_records)
        save_cols = [c for c in df_runs.columns if c != "_status"]
        df_runs[save_cols].to_csv(runs_path, index=False)
        df_mean = df_runs.groupby(["Model", "Sampler"])[METRIC_COLS].mean().reset_index()
        df_mean.to_csv(mean_path, index=False)
        _save_tables(df_mean, bench)
        _plot_figure2(df_mean)
        _print_comparison(df_mean, bench)

    print("\n" + "=" * 70)
    print(f" ALL DONE in {elapsed:.1f}s  |  results/ updated with tables and curves")
    print("=" * 70)


if __name__ == "__main__":
    main()
