import pandas as pd
import numpy as np
from tabulate import tabulate
from src.config import PAPER_TABLE2_BENCHMARKS, RESULTS_DIR


def compare_with_paper_benchmarks(results_csv_path=None):
    if results_csv_path is None:
        results_csv_path = RESULTS_DIR / "mean_benchmark_results.csv"

    if not results_csv_path.exists():
        print(f"Results file {results_csv_path} does not exist yet. Please run the benchmark first.")
        return

    df_ours = pd.read_csv(results_csv_path)

    comparison_rows = []

    for _, row in df_ours.iterrows():
        model = row["Model"]
        sampler = row["Sampler"]
        key = (model, sampler)

        if key in PAPER_TABLE2_BENCHMARKS:
            target = PAPER_TABLE2_BENCHMARKS[key]
            comparison_rows.append({
                "Model": model,
                "Sampler": sampler,
                "Paper_Acc": target["Accuracy"],
                "Our_Acc": round(row["Accuracy"], 3),
                "Paper_Sens": target["Sensitivity"],
                "Our_Sens": round(row["Sensitivity"], 3),
                "Paper_Spec": target["Specificity"],
                "Our_Spec": round(row["Specificity"], 4),
                "Paper_F1": target["F1"],
                "Our_F1": round(row["F1"], 3),
                "Paper_TCS": target["TCS"],
                "Our_TCS": round(row["TCS"], 2),
                "Paper_ACM": target["ACM"],
                "Our_ACM": round(row["ACM"], 2),
                "TCS_Diff": round(row["TCS"] - target["TCS"], 2),
            })

    df_comp = pd.DataFrame(comparison_rows)
    print("\n" + "=" * 80)
    print("REPLICATION BENCHMARK COMPARISON AGAINST PUBLISHED TABLE 2")
    print("=" * 80)
    print(tabulate(df_comp, headers="keys", tablefmt="pipe", showindex=False))

    summary_file = RESULTS_DIR / "replication_comparison_table.csv"
    df_comp.to_csv(summary_file, index=False)
    print(f"\nDetailed comparison saved to: {summary_file}")
    return df_comp


if __name__ == "__main__":
    compare_with_paper_benchmarks()
