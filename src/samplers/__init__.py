from src.samplers.standard_samplers import (
    StandardSMOTE,
    StandardADASYN,
    StandardBorderlineSMOTE,
    StandardSMOTEENN,
    StandardSMOTETomek,
)
from src.samplers.cost_samplers import (
    CostSensitiveSMOTE,
    CostSensitiveADASYN,
    CostSensitiveBorderlineSMOTE,
    CostSensitiveSMOTEENN,
    CostSensitiveSMOTETomek,
)

SAMPLER_REGISTRY = {
    "SMOTE": StandardSMOTE,
    "Cost-sensitive SMOTE": CostSensitiveSMOTE,
    "ADASYN": StandardADASYN,
    "Cost-sensitive ADASYN": CostSensitiveADASYN,
    "Borderline-SMOTE": StandardBorderlineSMOTE,
    "Cost-sensitive Borderline-SMOTE": CostSensitiveBorderlineSMOTE,
    "SMOTE-ENN": StandardSMOTEENN,
    "Cost-sensitive SMOTE-ENN": CostSensitiveSMOTEENN,
    "SMOTE-Tomek": StandardSMOTETomek,
    "Cost-sensitive SMOTE-Tomek": CostSensitiveSMOTETomek,
}


def get_sampler(sampler_name, random_state=42):
    """
    Factory function returning the sampler instance by name.
    """
    if sampler_name not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sampler: {sampler_name}. Available: {list(SAMPLER_REGISTRY.keys())}"
        )
    return SAMPLER_REGISTRY[sampler_name](random_state=random_state)
