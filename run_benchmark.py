import argparse
import sys
from src.data_loader import prepare_train_test_split
from src.evaluator import run_full_benchmark
from src.visualizer import plot_figure_2_tcs_acm, generate_roc_curves_for_model
from src.config import MODEL_CONFIGS, OVERSAMPLER_NAMES, SEEDS, RESULTS_DIR
from compare_results import compare_with_paper_benchmarks


def main():
    parser = argparse.ArgumentParser(
        description="Replicate 'Performance Analysis of Cost-Sensitive Oversampling Mechanisms for Credit Card Fraud Detection'"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        help="List of models to benchmark. Available: " + ", ".join(MODEL_CONFIGS.keys()),
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=OVERSAMPLER_NAMES,
        help="List of samplers to test.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds to evaluate over (default: 42). Paper uses 3 repeats.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate Figure 2 bar plots and Figures 3-7 ROC curves after benchmarking.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison against paper Table 2 after benchmarking.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Runs a fast verification test (Logistic Regression on SMOTE and Cost-sensitive SMOTE).",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("REPLICATION PIPELINE: COST-SENSITIVE OVERSAMPLING FOR FRAUD DETECTION")
    print("=" * 70)

    # 1. Load and prepare dataset
    print("\n[Step 1/4] Loading and preparing European Credit Card dataset...")
    data = prepare_train_test_split()
    print(f"Data ready. Train shape: {data['X_train'].shape}, Test shape: {data['X_test'].shape}")
    print(f"Train frauds: {data['y_train'].sum()}, Test frauds: {data['y_test'].sum()}")

    # Determine models and samplers to run
    if args.quick_test:
        models_to_run = ["Logistic Regression"]
        samplers_to_run = ["SMOTE", "Cost-sensitive SMOTE"]
        seeds_to_run = [42]
        print("\n*** Quick Test Mode Activated ***")
    else:
        models_to_run = args.models
        samplers_to_run = args.samplers
        seeds_to_run = args.seeds

    print(f"\n[Step 2/4] Running experiments:")
    print(f"Models ({len(models_to_run)}): {models_to_run}")
    print(f"Samplers ({len(samplers_to_run)}): {samplers_to_run}")
    print(f"Seeds: {seeds_to_run}")

    # 2. Run benchmark
    df_mean = run_full_benchmark(
        data=data,
        models=models_to_run,
        samplers=samplers_to_run,
        seeds=seeds_to_run,
        verbose=True,
    )

    # 3. Visualizations
    if args.plot:
        print("\n[Step 3/4] Generating paper figures...")
        plot_figure_2_tcs_acm(df_mean)
        for model_name in models_to_run:
            print(f"Generating ROC curves for {model_name}...")
            generate_roc_curves_for_model(model_name, data, samplers=samplers_to_run)
        print("All plots generated successfully!")
    else:
        print("\n[Step 3/4] Plotting skipped (use --plot to generate).")

    # 4. Compare with paper Table 2
    if args.compare or args.quick_test:
        print("\n[Step 4/4] Comparing with published Table 2 benchmarks...")
        compare_with_paper_benchmarks()

    print("\nExecution completed successfully!")


if __name__ == "__main__":
    main()
