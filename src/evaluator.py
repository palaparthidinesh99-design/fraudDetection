import time
import os
import numpy as np
import pandas as pd
from src.samplers import get_sampler
from src.models import get_model
from src.metrics import calculate_metrics
from src.config import SEEDS, MODEL_CONFIGS, OVERSAMPLER_NAMES, RESULTS_DIR


def evaluate_single_combination(
    model_name: str,
    sampler_name: str,
    data: dict,
    random_state: int = 42,
    return_predictions: bool = False,
):
    """
    Executes a single (model, sampler, seed) training & evaluation run.
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    amounts_train = data["amounts_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    amounts_test = data["amounts_test"]

    # 1. Resample training dataset
    sampler = get_sampler(sampler_name, random_state=random_state)
    start_resample = time.time()
    X_train_res, y_train_res = sampler.fit_resample(
        X_train, y_train, amounts=amounts_train
    )
    resample_time = time.time() - start_resample

    # 2. Train model
    model = get_model(model_name, random_state=random_state)
    start_train = time.time()
    model.fit(X_train_res, y_train_res)
    train_time = time.time() - start_train

    # 3. Predict on test set
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-df_vals))

    # 4. Compute metrics
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

    if return_predictions:
        return metrics, y_pred, y_prob
    return metrics


def run_full_benchmark(
    data: dict,
    models: list = None,
    samplers: list = None,
    seeds: list = None,
    verbose: bool = True,
):
    """
    Runs benchmark for specified models and samplers across specified random seeds.
    Saves results incrementally to CSV after every single run so no progress is ever lost.
    """
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    if samplers is None:
        samplers = OVERSAMPLER_NAMES
    if seeds is None:
        seeds = [42]

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
        print(f"Benchmark starting: {len(models)} models x {len(samplers)} samplers x {len(seeds)} seeds = {total_runs} total runs.")
        if len(existing_keys) > 0:
            print(f"Found {len(existing_keys)} pre-existing completed runs in {runs_path}. Skipping duplicates.")

    metric_cols = ["Accuracy", "Specificity", "Sensitivity", "F1", "TCS", "ACM", "AUC"]

    for m_idx, model_name in enumerate(models):
        for s_idx, sampler_name in enumerate(samplers):
            for seed in seeds:
                current_run += 1
                key = (model_name, sampler_name, int(seed))
                if key in existing_keys:
                    if verbose:
                        print(f"[{current_run}/{total_runs}] Skipping {model_name} with {sampler_name} (seed={seed}) [Already Completed]")
                    continue

                if verbose:
                    print(f"[{current_run}/{total_runs}] Running {model_name} with {sampler_name} (seed={seed})...", flush=True)

                res = evaluate_single_combination(
                    model_name=model_name,
                    sampler_name=sampler_name,
                    data=data,
                    random_state=seed,
                )
                all_run_records.append(res)
                existing_keys.add(key)

                if verbose:
                    print(f"   -> Result: F1={res['F1']:.4f}, TCS={res['TCS']:.2f}, ACM={res['ACM']:.2f} (Time: {res['TrainTimeSec']}s)", flush=True)

                # Persist to disk incrementally immediately
                df_current = pd.DataFrame(all_run_records)
                df_current.to_csv(runs_path, index=False)

                # Update mean summary file incrementally
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
        print(f"\nAll benchmark runs completed and saved!")
        print(f"Raw results: {runs_path}")
        print(f"Mean results: {mean_path}")

    return df_mean
