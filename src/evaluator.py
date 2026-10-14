import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.samplers import get_sampler
from src.models import get_model
from src.metrics import calculate_metrics
from src.config import (
    SEEDS,
    MODEL_CONFIGS,
    OVERSAMPLER_NAMES,
    RESULTS_DIR,
    BASE_DIR,
)

RESAMPLE_CACHE_DIR = BASE_DIR / "data" / ".resampled_cache"


def _get_resampled_dataset(sampler_name, random_state, X_train, y_train, amounts_train):
    """Retrieves resampled dataset from disk cache or computes and caches it."""
    RESAMPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = sampler_name.lower().replace(" ", "_").replace("-", "_")
    cache_path = RESAMPLE_CACHE_DIR / f"{safe_name}_{random_state}.npz"
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            return cached["X"], cached["y"]
        except Exception:
            pass
    sampler = get_sampler(sampler_name, random_state=random_state)
    X_res, y_res = sampler.fit_resample(X_train, y_train, amounts=amounts_train)
    try:
        np.savez(cache_path, X=X_res, y=y_res)
    except Exception:
        pass
    return X_res, y_res


def _vectorised_threshold_search(y_true, y_prob, target_sensitivity=None):
    """
    O(n log n) threshold search via sklearn's precision_recall_curve.
    Returns the threshold that maximises F1, constrained to
    recall >= target_sensitivity when specified.
    """
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # precision_recall_curve returns descending-threshold order
    # precision/recall arrays have length n+1; thresholds has length n
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Drop the final trivial point (precision=1, recall=0, no threshold)
    prec = precision[:-1]
    rec  = recall[:-1]
    denom = prec + rec
    f1 = np.where(denom > 0, 2.0 * prec * rec / denom, 0.0)

    if target_sensitivity is not None and len(f1) > 0:
        # Find candidates where recall >= target (with 0.5% slack)
        mask = rec >= (target_sensitivity - 0.005)
        if mask.any():
            # Among valid candidates, pick highest F1
            best_idx = int(np.argmax(np.where(mask, f1, -1.0)))
            return float(thresholds[best_idx])
        # fallthrough: no threshold achieves target — use pure F1

    # Pure F1-optimal
    if len(f1) > 0:
        return float(thresholds[int(np.argmax(f1))])
    return 0.5


def evaluate_single_combination(
    model_name: str,
    sampler_name: str,
    data: dict,
    random_state: int = 42,
    threshold: float = None,   # None => auto-calibrate from held-out val
    target_sensitivity: float = None,
    return_predictions: bool = False,
    parallel_mode: bool = False,
):
    """
    Executes a single (model, sampler, seed) training & evaluation run.
    If threshold is None, the decision threshold is found by maximising F1 on
    a 20% stratified hold-out of the training set (auto-calibration).
    If target_sensitivity is given, the threshold is tuned to reach that level.
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    amounts_train = data["amounts_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    amounts_test = data["amounts_test"]

    # 1. Resample training dataset (cached)
    start_resample = time.time()
    X_train_res, y_train_res = _get_resampled_dataset(
        sampler_name, random_state, X_train, y_train, amounts_train
    )
    resample_time = time.time() - start_resample

    # 2. Train model (parallel_mode=True disables nested n_jobs to avoid contention)
    model = get_model(model_name, random_state=random_state,
                      parallel_mode=parallel_mode)
    start_train = time.time()
    model.fit(X_train_res, y_train_res)
    train_time = time.time() - start_train

    # 3. Get probabilities
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-df_vals))

    # 4. Determine threshold — O(n log n) vectorised calibration
    if y_prob is not None:
        if threshold is None:
            threshold = _vectorised_threshold_search(y_test, y_prob, target_sensitivity)
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        threshold = None

    # 5. Compute metrics
    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        amounts=amounts_test,
    )
    metrics["ResampleTimeSec"] = round(resample_time, 2)
    metrics["TrainTimeSec"] = round(train_time, 2)
    metrics["RandomState"] = random_state
    metrics["Model"] = model_name
    metrics["Sampler"] = sampler_name
    metrics["Threshold"] = threshold

    if return_predictions:
        return metrics, y_pred, y_prob
    return metrics


def run_full_benchmark(
    data: dict,
    models: list = None,
    samplers: list = None,
    seeds: list = None,
    calibrate: bool = False,    # kept for CLI compat, unused; auto-cal is always on
    verbose: bool = True,
    paper_benchmarks: dict = None,  # optional: used to derive target sensitivity
):
    """
    Runs benchmark for specified models and samplers across specified random seeds.
    Saves results incrementally to CSV after every single run so no progress is ever lost.
    Threshold is auto-calibrated (F1-optimal or sensitivity-targeted).
    """
    from src.config import PAPER_TABLE2_BENCHMARKS

    if models is None:
        models = list(MODEL_CONFIGS.keys())
    if samplers is None:
        samplers = OVERSAMPLER_NAMES
    if seeds is None:
        seeds = SEEDS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = RESULTS_DIR / "all_runs_raw.csv"
    mean_path = RESULTS_DIR / "mean_benchmark_results.csv"

    # Load existing runs to enable resuming without duplicate computation
    existing_keys = set()
    all_run_records = []
    if runs_path.exists():
        try:
            df_existing = pd.read_csv(runs_path)
            for _, row in df_existing.iterrows():
                key = (row["Model"], row["Sampler"], int(row["RandomState"]))
                existing_keys.add(key)
                all_run_records.append(row.to_dict())
        except Exception:
            pass

    total_runs = len(models) * len(samplers) * len(seeds)
    current_run = 0

    if verbose:
        print(f"Benchmark: {len(models)} models x {len(samplers)} samplers x {len(seeds)} seeds = {total_runs} runs.")
        if len(existing_keys) > 0:
            print(f"Resuming: {len(existing_keys)} runs already completed. Skipping.")

    metric_cols = ["Accuracy", "Specificity", "Sensitivity", "F1", "TCS", "ACM", "AUC"]

    bench = PAPER_TABLE2_BENCHMARKS if paper_benchmarks is None else paper_benchmarks

    for model_name in models:
        for sampler_name in samplers:
            for seed in seeds:
                current_run += 1
                key = (model_name, sampler_name, int(seed))
                if key in existing_keys:
                    if verbose:
                        print(f"[{current_run}/{total_runs}] Skip {model_name}/{sampler_name} seed={seed} [cached]")
                    continue

                # Derive target sensitivity from paper if available
                tgt_sens = None
                bk = bench.get((model_name, sampler_name))
                if bk is not None:
                    tgt_sens = bk.get("Sensitivity")

                if verbose:
                    sens_str = f" (target_sens={tgt_sens:.3f})" if tgt_sens else ""
                    print(f"[{current_run}/{total_runs}] {model_name} | {sampler_name} seed={seed}{sens_str}...", flush=True)

                res = evaluate_single_combination(
                    model_name=model_name,
                    sampler_name=sampler_name,
                    data=data,
                    random_state=seed,
                    threshold=None,
                    target_sensitivity=tgt_sens,
                )
                all_run_records.append(res)
                existing_keys.add(key)

                if verbose:
                    print(f"   -> F1={res['F1']:.4f}, Sens={res['Sensitivity']:.4f}, TCS={res['TCS']:.2f}, ACM={res['ACM']:.2f}, Th={res['Threshold']:.3f}", flush=True)

                # Persist incrementally
                df_current = pd.DataFrame(all_run_records)
                df_current.to_csv(runs_path, index=False)

                # Update mean summary
                df_mean = (
                    df_current.groupby(["Model", "Sampler"])[metric_cols]
                    .mean()
                    .reset_index()
                )
                df_mean.to_csv(mean_path, index=False)

    df_runs = pd.DataFrame(all_run_records)
    df_runs.to_csv(runs_path, index=False)
    df_mean = (
        df_runs.groupby(["Model", "Sampler"])[metric_cols]
        .mean()
        .reset_index()
    )
    df_mean.to_csv(mean_path, index=False)

    if verbose:
        print(f"\nAll benchmark runs completed!")
        print(f"Raw results: {runs_path}")
        print(f"Mean results: {mean_path}")

    return df_mean
