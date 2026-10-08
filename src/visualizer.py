import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from src.config import PLOTS_DIR, OVERSAMPLER_NAMES, MODEL_CONFIGS
from src.samplers import get_sampler
from src.models import get_model


def plot_figure_2_tcs_acm(df_mean, output_dir=PLOTS_DIR):
    """
    Reproduces Figure 2 from the paper:
    Bar charts of oversampling method vs TCS and ACM (on log scale) for each model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    models = df_mean["Model"].unique()

    for model_name in models:
        df_m = df_mean[df_mean["Model"] == model_name].copy()
        # Set categorical order for consistent plotting
        df_m["Sampler"] = pd.Categorical(df_m["Sampler"], categories=OVERSAMPLER_NAMES, ordered=True)
        df_m = df_m.sort_values("Sampler")

        x = np.arange(len(df_m))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, df_m["TCS"], width, label="TCS", color="#3498db")
        rects2 = ax.bar(x + width/2, df_m["ACM"], width, label="ACM", color="#e74c3c")

        ax.set_ylabel("Log Scale", fontsize=12)
        ax.set_yscale("log")
        ax.set_title(f"({model_name}) TCS and ACM plots", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df_m["Sampler"], rotation=45, ha="right", fontsize=10)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        filename = f"figure_2_{model_name.lower().replace(' ', '_')}_tcs_acm.png"
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved Figure 2 plot for {model_name} to: {save_path}")


def generate_roc_curves_for_model(model_name, data, samplers=OVERSAMPLER_NAMES, output_dir=PLOTS_DIR, random_state=42):
    """
    Reproduces Figures 3-7 from the paper:
    ROC curves for each classifier comparing standard vs cost-sensitive oversampling methods.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))

    X_train, y_train, amounts_train = data["X_train"], data["y_train"], data["amounts_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    for sampler_name in samplers:
        sampler = get_sampler(sampler_name, random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train, amounts=amounts_train)

        clf = get_model(model_name, random_state=random_state)
        clf.fit(X_res, y_res)

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            df_vals = clf.decision_function(X_test)
            y_prob = 1.0 / (1.0 + np.exp(-df_vals))
        else:
            y_prob = clf.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        is_cs = "Cost-sensitive" in sampler_name
        linestyle = "-" if is_cs else "--"
        plt.plot(fpr, tpr, label=f"{sampler_name} (AUC={roc_auc:.3f})", linestyle=linestyle, lw=1.5)

    # Reference diagonal
    plt.plot([0, 1], [0, 1], color="grey", linestyle=":")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title(f"ROC Curves for {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)

    filename = f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve plot for {model_name} to: {save_path}")
