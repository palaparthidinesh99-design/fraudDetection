import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from src.samplers.base import BaseOversampler


class StandardSMOTE(BaseOversampler):
    def __init__(self, random_state=42, k_neighbors=5):
        super().__init__(random_state=random_state)
        self.sampler = SMOTE(
            sampling_strategy=1.0,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )

    def fit_resample(self, X, y, amounts=None):
        return self.sampler.fit_resample(X, y)


class StandardADASYN(BaseOversampler):
    def __init__(self, random_state=42, n_neighbors=5):
        super().__init__(random_state=random_state)
        self.sampler = ADASYN(
            sampling_strategy=1.0,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )

    def fit_resample(self, X, y, amounts=None):
        return self.sampler.fit_resample(X, y)


class StandardBorderlineSMOTE(BaseOversampler):
    def __init__(self, random_state=42, k_neighbors=5, m_neighbors=10):
        super().__init__(random_state=random_state)
        self.sampler = BorderlineSMOTE(
            sampling_strategy=1.0,
            k_neighbors=k_neighbors,
            m_neighbors=m_neighbors,
            kind="borderline-1",
            random_state=random_state,
        )

    def fit_resample(self, X, y, amounts=None):
        return self.sampler.fit_resample(X, y)


class StandardSMOTEENN(BaseOversampler):
    def __init__(self, random_state=42, smote_k_neighbors=5):
        super().__init__(random_state=random_state)
        smote = SMOTE(
            sampling_strategy=1.0,
            k_neighbors=smote_k_neighbors,
            random_state=random_state,
        )
        enn = EditedNearestNeighbours(n_neighbors=3, n_jobs=-1)
        self.sampler = SMOTEENN(
            smote=smote,
            enn=enn,
            random_state=random_state,
        )

    def fit_resample(self, X, y, amounts=None):
        return self.sampler.fit_resample(X, y)


class StandardSMOTETomek(BaseOversampler):
    def __init__(self, random_state=42, smote_k_neighbors=5):
        super().__init__(random_state=random_state)
        smote = SMOTE(
            sampling_strategy=1.0,
            k_neighbors=smote_k_neighbors,
            random_state=random_state,
        )
        tomek = TomekLinks(n_jobs=-1)
        self.sampler = SMOTETomek(
            smote=smote,
            tomek=tomek,
            random_state=random_state,
        )

    def fit_resample(self, X, y, amounts=None):
        return self.sampler.fit_resample(X, y)
