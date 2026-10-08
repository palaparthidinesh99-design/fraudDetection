import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

DATASET_PATH = DATA_DIR / "creditcard.csv"

# Preprocessing & Split
TEST_SIZE = 0.30
RANDOM_STATE = 42
N_REPEATS = 3  # Paper: 'Each experiment was repeated three times and mean results are reported.'

# Seed list for the 3 experimental runs
SEEDS = [42, 100, 2024]

# Model Hyperparameters as specified in Table 1 of the paper
MODEL_CONFIGS = {
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42,
        "n_jobs": -1,
    },
    "XGBoost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": -1,
    },
    "LightGBM": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    },
    "MLP": {
        "hidden_layer_sizes": (64,),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 100,
        "random_state": 42,
    },
    "Logistic Regression": {
        "solver": "lbfgs",
        "random_state": 42,
        "max_iter": 1000,
    },
}

# Oversamplers list
OVERSAMPLER_NAMES = [
    "SMOTE",
    "Cost-sensitive SMOTE",
    "ADASYN",
    "Cost-sensitive ADASYN",
    "Borderline-SMOTE",
    "Cost-sensitive Borderline-SMOTE",
    "SMOTE-ENN",
    "Cost-sensitive SMOTE-ENN",
    "SMOTE-Tomek",
    "Cost-sensitive SMOTE-Tomek",
]

# Paper Ground Truth from Table 2 for verification
PAPER_TABLE2_BENCHMARKS = {
    ("Random Forest", "SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9998, "Sensitivity": 0.797, "F1": 0.834, "TCS": 12331.56, "ACM": 148.88},
    ("Random Forest", "Cost-sensitive SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9999, "Sensitivity": 0.750, "F1": 0.841, "TCS": 9941.71, "ACM": 223.22},
    ("Random Forest", "ADASYN"): {"Accuracy": 0.999, "Specificity": 0.9998, "Sensitivity": 0.784, "F1": 0.820, "TCS": 12329.56, "ACM": 137.28},
    ("Random Forest", "Cost-sensitive ADASYN"): {"Accuracy": 0.999, "Specificity": 0.9999, "Sensitivity": 0.736, "F1": 0.838, "TCS": 11597.06, "ACM": 183.76},
    ("Random Forest", "Borderline-SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9998, "Sensitivity": 0.777, "F1": 0.833, "TCS": 10501.84, "ACM": 191.81},
    ("Random Forest", "Cost-sensitive Borderline-SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9999, "Sensitivity": 0.743, "F1": 0.846, "TCS": 12594.43, "ACM": 106.34},
    ("Random Forest", "SMOTE-ENN"): {"Accuracy": 0.999, "Specificity": 0.9996, "Sensitivity": 0.811, "F1": 0.805, "TCS": 11465.27, "ACM": 191.81},
    ("Random Forest", "Cost-sensitive SMOTE-ENN"): {"Accuracy": 0.999, "Specificity": 0.9998, "Sensitivity": 0.757, "F1": 0.815, "TCS": 11465.27, "ACM": 196.22},
    ("Random Forest", "SMOTE-Tomek"): {"Accuracy": 0.999, "Specificity": 0.9998, "Sensitivity": 0.804, "F1": 0.835, "TCS": 12332.56, "ACM": 148.88},
    ("Random Forest", "Cost-sensitive SMOTE-Tomek"): {"Accuracy": 0.999, "Specificity": 0.9999, "Sensitivity": 0.750, "F1": 0.841, "TCS": 9941.71, "ACM": 223.22},

    ("XGBoost", "SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9996, "Sensitivity": 0.831, "F1": 0.807, "TCS": 12778.31, "ACM": 111.31},
    ("XGBoost", "Cost-sensitive SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9999, "Sensitivity": 0.777, "F1": 0.849, "TCS": 12147.08, "ACM": 174.95},
    ("XGBoost", "ADASYN"): {"Accuracy": 0.999, "Specificity": 0.9996, "Sensitivity": 0.824, "F1": 0.797, "TCS": 12681.11, "ACM": 107.53},
    ("XGBoost", "Cost-sensitive ADASYN"): {"Accuracy": 0.999, "Specificity": 0.9999, "Sensitivity": 0.777, "F1": 0.855, "TCS": 12147.08, "ACM": 183.87},
    ("XGBoost", "Borderline-SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9997, "Sensitivity": 0.797, "F1": 0.817, "TCS": 12471.07, "ACM": 129.51},
    ("XGBoost", "Cost-sensitive Borderline-SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9999, "Sensitivity": 0.770, "F1": 0.841, "TCS": 12146.08, "ACM": 166.86},
    ("XGBoost", "SMOTE-ENN"): {"Accuracy": 0.999, "Specificity": 0.9994, "Sensitivity": 0.824, "F1": 0.770, "TCS": 12433.85, "ACM": 132.65},
    ("XGBoost", "Cost-sensitive SMOTE-ENN"): {"Accuracy": 0.999, "Specificity": 0.9998, "Sensitivity": 0.784, "F1": 0.817, "TCS": 12980.10, "ACM": 70.12},
    ("XGBoost", "SMOTE-Tomek"): {"Accuracy": 0.999, "Specificity": 0.9996, "Sensitivity": 0.824, "F1": 0.805, "TCS": 12777.31, "ACM": 111.31},
    ("XGBoost", "Cost-sensitive SMOTE-Tomek"): {"Accuracy": 0.999, "Specificity": 0.9999, "Sensitivity": 0.777, "F1": 0.849, "TCS": 12147.08, "ACM": 174.95},

    ("LightGBM", "SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9992, "Sensitivity": 0.811, "F1": 0.714, "TCS": 12252.59, "ACM": 74.24},
    ("LightGBM", "Cost-sensitive SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9998, "Sensitivity": 0.770, "F1": 0.823, "TCS": 12233.49, "ACM": 144.76},
    ("LightGBM", "ADASYN"): {"Accuracy": 0.998, "Specificity": 0.9989, "Sensitivity": 0.784, "F1": 0.643, "TCS": 10958.55, "ACM": 65.51},
    ("LightGBM", "Cost-sensitive ADASYN"): {"Accuracy": 0.999, "Specificity": 0.9999, "Sensitivity": 0.777, "F1": 0.836, "TCS": 12321.95, "ACM": 155.60},
    ("LightGBM", "Borderline-SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9997, "Sensitivity": 0.797, "F1": 0.808, "TCS": 10821.09, "ACM": 152.09},
    ("LightGBM", "Cost-sensitive Borderline-SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9998, "Sensitivity": 0.777, "F1": 0.816, "TCS": 12235.49, "ACM": 136.45},
    ("LightGBM", "SMOTE-ENN"): {"Accuracy": 0.999, "Specificity": 0.9991, "Sensitivity": 0.804, "F1": 0.688, "TCS": 11255.63, "ACM": 75.33},
    ("LightGBM", "Cost-sensitive SMOTE-ENN"): {"Accuracy": 0.999, "Specificity": 0.9998, "Sensitivity": 0.797, "F1": 0.822, "TCS": 12735.85, "ACM": 49.13},
    ("LightGBM", "SMOTE-Tomek"): {"Accuracy": 0.999, "Specificity": 0.9993, "Sensitivity": 0.818, "F1": 0.736, "TCS": 12487.61, "ACM": 135.24},
    ("LightGBM", "Cost-sensitive SMOTE-Tomek"): {"Accuracy": 0.999, "Specificity": 0.9998, "Sensitivity": 0.770, "F1": 0.823, "TCS": 12233.49, "ACM": 144.76},

    ("MLP", "SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9989, "Sensitivity": 0.818, "F1": 0.663, "TCS": 13555.58, "ACM": 47.58},
    ("MLP", "Cost-sensitive SMOTE"): {"Accuracy": 0.998, "Specificity": 0.9983, "Sensitivity": 0.804, "F1": 0.575, "TCS": 12982.90, "ACM": 36.80},
    ("MLP", "ADASYN"): {"Accuracy": 0.999, "Specificity": 0.9989, "Sensitivity": 0.818, "F1": 0.661, "TCS": 12755.20, "ACM": 53.66},
    ("MLP", "Cost-sensitive ADASYN"): {"Accuracy": 0.998, "Specificity": 0.9982, "Sensitivity": 0.838, "F1": 0.582, "TCS": 14057.91, "ACM": 30.38},
    ("MLP", "Borderline-SMOTE"): {"Accuracy": 0.999, "Specificity": 0.9994, "Sensitivity": 0.811, "F1": 0.757, "TCS": 12781.80, "ACM": 85.44},
    ("MLP", "Cost-sensitive Borderline-SMOTE"): {"Accuracy": 0.998, "Specificity": 0.9987, "Sensitivity": 0.804, "F1": 0.631, "TCS": 13691.95, "ACM": 41.22},
    ("MLP", "SMOTE-ENN"): {"Accuracy": 0.997, "Specificity": 0.9975, "Sensitivity": 0.811, "F1": 0.496, "TCS": 13298.21, "ACM": 25.53},
    ("MLP", "Cost-sensitive SMOTE-ENN"): {"Accuracy": 0.998, "Specificity": 0.9981, "Sensitivity": 0.818, "F1": 0.558, "TCS": 15114.60, "ACM": 22.72},
    ("MLP", "SMOTE-Tomek"): {"Accuracy": 0.995, "Specificity": 0.9957, "Sensitivity": 0.845, "F1": 0.393, "TCS": 12982.90, "ACM": 36.80},
    ("MLP", "Cost-sensitive SMOTE-Tomek"): {"Accuracy": 0.998, "Specificity": 0.9983, "Sensitivity": 0.804, "F1": 0.575, "TCS": 15921.86, "ACM": 9.72},

    ("Logistic Regression", "SMOTE"): {"Accuracy": 0.983, "Specificity": 0.9828, "Sensitivity": 0.858, "F1": 0.145, "TCS": 13740.50, "ACM": 8.72},
    ("Logistic Regression", "Cost-sensitive SMOTE"): {"Accuracy": 0.986, "Specificity": 0.9866, "Sensitivity": 0.865, "F1": 0.181, "TCS": 15431.77, "ACM": 9.33},
    ("Logistic Regression", "ADASYN"): {"Accuracy": 0.957, "Specificity": 0.9572, "Sensitivity": 0.865, "F1": 0.065, "TCS": 14392.03, "ACM": 11.32},
    ("Logistic Regression", "Cost-sensitive ADASYN"): {"Accuracy": 0.986, "Specificity": 0.9866, "Sensitivity": 0.865, "F1": 0.181, "TCS": 15431.77, "ACM": 1.34},
    ("Logistic Regression", "Borderline-SMOTE"): {"Accuracy": 0.994, "Specificity": 0.9940, "Sensitivity": 0.831, "F1": 0.315, "TCS": 12593.51, "ACM": 13.51},
    ("Logistic Regression", "Cost-sensitive Borderline-SMOTE"): {"Accuracy": 0.986, "Specificity": 0.9867, "Sensitivity": 0.865, "F1": 0.181, "TCS": 15431.77, "ACM": 14.80},
    ("Logistic Regression", "SMOTE-ENN"): {"Accuracy": 0.982, "Specificity": 0.9821, "Sensitivity": 0.858, "F1": 0.141, "TCS": 13740.50, "ACM": 14.59},
    ("Logistic Regression", "Cost-sensitive SMOTE-ENN"): {"Accuracy": 0.986, "Specificity": 0.9866, "Sensitivity": 0.865, "F1": 0.180, "TCS": 15431.77, "ACM": 4.42},
    ("Logistic Regression", "SMOTE-Tomek"): {"Accuracy": 0.983, "Specificity": 0.9828, "Sensitivity": 0.858, "F1": 0.146, "TCS": 13740.50, "ACM": 8.73},
    ("Logistic Regression", "Cost-sensitive SMOTE-Tomek"): {"Accuracy": 0.986, "Specificity": 0.9866, "Sensitivity": 0.865, "F1": 0.181, "TCS": 15431.77, "ACM": 5.33},
}
