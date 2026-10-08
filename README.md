# Replication: Performance Analysis of Cost-Sensitive Oversampling Mechanisms for Credit Card Fraud Detection

Complete, modular implementation replicating the IEEE Access (2025) study:

> **Paper**: Emmanuel Ileberi and Yanxia Sun, *"Performance Analysis of Cost-Sensitive Oversampling Mechanisms for Credit Card Fraud Detection,"* *IEEE Access*, vol. 13, pp. 202655–202664, 2025. [DOI: 10.1109/ACCESS.2025.3637132](https://doi.org/10.1109/ACCESS.2025.3637132)

---

## 1. Project Overview

Credit card fraud datasets suffer from extreme class imbalance (~0.17% fraud rate) and heterogeneous financial costs. Standard oversampling techniques (SMOTE, ADASYN, Borderline-SMOTE, SMOTE-ENN, SMOTE-Tomek) treat every fraudulent transaction uniformly.

This project implements the paper's data-level, classifier-agnostic framework that weights minority sample generation directly by normalized transaction amount:

$$w_i = \frac{c_i}{\sum_{j \in S_{\min}} c_j}$$

$$n_i = \alpha \cdot w_i \cdot |S_{\text{maj}}|$$

The framework is systematically benchmarked across **5 machine learning classifiers** and **10 oversampling techniques** (5 standard + 5 cost-sensitive variants).

---

## 2. Repository Structure

```
ml/
├── data/
│   └── creditcard.csv            # European credit card dataset (284,807 transactions)
├── src/
│   ├── __init__.py
│   ├── config.py                 # Hyperparameters (Table 1), benchmarks (Table 2), paths
│   ├── data_loader.py            # Stratified 70/30 split and 30-feature standardizer
│   ├── metrics.py                # Acc, Sens, Spec, F1, TCS, ACM, ROC-AUC
│   ├── models.py                 # Classifier factories (Table 1 specs)
│   ├── samplers/
│   │   ├── __init__.py           # Unified sampler factory
│   │   ├── base.py               # Abstract base oversampler
│   │   ├── standard_samplers.py  # SMOTE, ADASYN, Borderline-SMOTE, SMOTE-ENN, SMOTE-Tomek
│   │   └── cost_samplers.py      # CS-SMOTE, CS-ADASYN, CS-Borderline, CS-ENN, CS-Tomek
│   ├── evaluator.py              # Single and multi-run experimental runner
│   └── visualizer.py             # Reproduces Figure 2 (TCS & ACM) and Figures 3-7 (ROC)
├── tests/
│   └── test_samplers.py          # Unit tests verifying all 10 oversamplers
├── results/                      # Generated CSVs and benchmark comparisons
├── plots/                        # Generated ROC curves and bar charts
├── compare_results.py            # Verification tool comparing against Table 2 ground truth
├── run_benchmark.py              # Main CLI execution entrypoint
├── requirements.txt              # Project dependencies
└── run.sh                        # One-command runner script
```

---

## 3. Classifiers & Table 1 Hyperparameters

| Classifier | Specified Hyperparameters |
| :--- | :--- |
| **Random Forest** | `n_estimators=100`, `max_depth=None`, `random_state=42` |
| **XGBoost** | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`, `eval_metric='logloss'` |
| **LightGBM** | `n_estimators=100`, `learning_rate=0.1`, `num_leaves=31`, `random_state=42`, `verbosity=-1` |
| **MLP** | `hidden_layer_sizes=(64,)`, `activation='relu'`, `solver='adam'`, `max_iter=100`, `random_state=42` |
| **Logistic Regression** | `penalty='l2'`, `solver='lbfgs'`, `random_state=42` |

---

## 4. Evaluation Metrics Formulation

- **Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
- **Sensitivity (Recall)**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
- **Specificity**: $\frac{\text{TN}}{\text{TN} + \text{FP}}$
- **F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Total Cost Savings (TCS)**:
  $$\text{TCS} = \sum_{i \in \text{FN}_{\text{baseline}}} a_i - \sum_{j \in \text{FN}_{\text{model}}} a_j = \sum_{k \in \text{TP}_{\text{model}}} a_k$$
- **Average Cost per Misclassification (ACM)**:
  $$\text{ACM} = \frac{\sum_{i \in \text{FN}} a_i + N_{\text{FP}} \cdot c_{\text{FP}}}{N_{\text{FN}} + N_{\text{FP}}} = \frac{\sum_{i \in \text{FN}} a_i}{N_{\text{FN}} + N_{\text{FP}}} \quad (c_{\text{FP}} = 0)$$

---

## 5. Quickstart & Execution

### 5.1 Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Run Unit Tests
```bash
python3 -m unittest discover tests
```

### 5.3 Quick Pipeline Verification
```bash
python3 run_benchmark.py --quick-test
```

### 5.4 Run Target Model Benchmarks
Run a single model across all 10 oversamplers:
```bash
python3 run_benchmark.py --models "Logistic Regression" --plot --compare
python3 run_benchmark.py --models "Random Forest" --plot --compare
python3 run_benchmark.py --models "XGBoost" --plot --compare
python3 run_benchmark.py --models "LightGBM" --plot --compare
python3 run_benchmark.py --models "MLP" --plot --compare
```

### 5.5 Run Full Benchmark Suite (5 Models $\times$ 10 Samplers $\times$ 3 Repeats)
```bash
python3 run_benchmark.py --seeds 42 100 2024 --plot --compare
```
