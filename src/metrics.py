import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_pred, y_prob=None, amounts=None):
    """
    Computes statistical and financial metrics according to the paper:
    - Accuracy: (TP + TN) / Total
    - Sensitivity (Recall): TP / (TP + FN)
    - Specificity: TN / (TN + FP)
    - F1-Score: Harmonic mean of precision and recall
    - Total Cost Savings (TCS): Reduction in financial loss relative to baseline of no detection.
        TCS = sum_{i in FN_baseline} a_i - sum_{j in FN_model} a_j
            = sum_{k in TP_model} a_k (total amount of successfully caught frauds)
    - Average Cost per Misclassification (ACM):
        ACM = (sum_{i in FN} a_i + N_FP * c_FP) / (N_FN + N_FP)
        where c_FP = 0 (false positive cost is negligible per paper Section V.A)
    - ROC-AUC: Area under the ROC curve (if y_prob provided)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Cost-based metrics
    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        # Fraud instances where true label is 1
        fraud_mask = (y_true == 1)
        tp_mask = (y_true == 1) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)

        total_fraud_cost_baseline = np.sum(amounts[fraud_mask])
        missed_fraud_cost_model = np.sum(amounts[fn_mask])

        # TCS: Saved fraud losses
        tcs = total_fraud_cost_baseline - missed_fraud_cost_model

        # ACM: Average cost per misclassification (FN + FP)
        num_misclassifications = fn + fp
        if num_misclassifications > 0:
            acm = missed_fraud_cost_model / num_misclassifications
        else:
            acm = 0.0
    else:
        tcs = np.nan
        acm = np.nan

    # ROC AUC
    auc = np.nan
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan

    return {
        "Accuracy": float(accuracy),
        "Specificity": float(specificity),
        "Sensitivity": float(sensitivity),
        "F1": float(f1),
        "TCS": float(tcs),
        "ACM": float(acm),
        "AUC": float(auc) if not np.isnan(auc) else np.nan,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }
