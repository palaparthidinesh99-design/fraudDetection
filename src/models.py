from src.config import MODEL_CONFIGS


def get_model(model_name, random_state=42):
    """
    Factory function instantiating classifiers with exact Table 1 hyperparameter configurations.
    Uses lazy importing so individual classifiers can be used independently.
    """
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    params = cfg.copy()
    if "random_state" in params:
        params["random_state"] = random_state

    if model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    elif model_name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(**params)
    elif model_name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Classifier {model_name} not implemented.")
