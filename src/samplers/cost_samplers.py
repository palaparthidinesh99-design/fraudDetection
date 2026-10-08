import numpy as np
from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from src.samplers.base import BaseOversampler


def allocate_samples(weights, total_samples):
    """
    Distributes total_samples proportionally to normalized weights,
    guaranteeing that the sum of allocated samples exactly equals total_samples.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0 or np.all(weights == 0):
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    raw_alloc = weights * total_samples
    int_alloc = np.floor(raw_alloc).astype(int)
    remainder = total_samples - int_alloc.sum()

    if remainder > 0:
        fractional_parts = raw_alloc - int_alloc
        top_indices = np.argsort(fractional_parts)[::-1][:remainder]
        int_alloc[top_indices] += 1

    return int_alloc


class CostSensitiveSMOTE(BaseOversampler):
    """
    Cost-Sensitive SMOTE according to Algorithm 1:
    The number of synthetic samples generated for each minority instance
    is proportional to its transaction amount (cost).
    """

    def __init__(self, random_state=42, k_neighbors=5):
        super().__init__(random_state=random_state)
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y, amounts=None):
        rng = np.random.RandomState(self.random_state)

        X_maj = X[y == 0]
        X_min = X[y == 1]
        n_maj = len(X_maj)
        n_min = len(X_min)

        if n_min >= n_maj:
            return X.copy(), y.copy()

        # Costs ci derived from transaction amounts
        if amounts is not None:
            amounts_min = amounts[y == 1]
            costs = np.maximum(amounts_min, 0.01)
        else:
            costs = np.ones(n_min, dtype=float)

        weights = costs / costs.sum()
        total_synthetic = n_maj - n_min
        allocations = allocate_samples(weights, total_synthetic)

        # Fit k-NN on minority class
        k = min(self.k_neighbors, n_min - 1)
        if k < 1:
            return X.copy(), y.copy()

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
        nn.fit(X_min)
        indices = nn.kneighbors(X_min, return_distance=False)[:, 1:]

        synthetic_samples = []
        for i in range(n_min):
            n_samples = allocations[i]
            if n_samples == 0:
                continue

            neighbor_choices = rng.choice(indices[i], size=n_samples, replace=True)
            lambdas = rng.uniform(0.0, 1.0, size=(n_samples, 1))
            diffs = X_min[neighbor_choices] - X_min[i]
            syn = X_min[i] + lambdas * diffs
            synthetic_samples.append(syn)

        if len(synthetic_samples) > 0:
            X_syn = np.vstack(synthetic_samples)
            y_syn = np.ones(len(X_syn), dtype=int)
            X_resampled = np.vstack([X, X_syn])
            y_resampled = np.concatenate([y, y_syn])
        else:
            X_resampled, y_resampled = X.copy(), y.copy()

        return X_resampled, y_resampled


class CostSensitiveADASYN(BaseOversampler):
    """
    Cost-Sensitive ADASYN:
    Paper Section IV: 'For ADASYN, rather than distributing synthetic samples
    according to local learning difficulty, we assign them according to cost,
    using w_i for each minority instance.'
    """

    def __init__(self, random_state=42, n_neighbors=5):
        super().__init__(random_state=random_state)
        self.n_neighbors = n_neighbors

    def fit_resample(self, X, y, amounts=None):
        cs_smote = CostSensitiveSMOTE(
            random_state=self.random_state,
            k_neighbors=self.n_neighbors,
        )
        return cs_smote.fit_resample(X, y, amounts=amounts)


class CostSensitiveBorderlineSMOTE(BaseOversampler):
    """
    Cost-Sensitive Borderline-SMOTE:
    Restricts interpolation to minority samples in the DANGER zone
    (where m/2 <= majority_neighbors < m), with sampling frequency
    within the danger zone proportional to instance cost.
    """

    def __init__(self, random_state=42, k_neighbors=5, m_neighbors=10):
        super().__init__(random_state=random_state)
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors

    def fit_resample(self, X, y, amounts=None):
        rng = np.random.RandomState(self.random_state)

        X_maj = X[y == 0]
        X_min = X[y == 1]
        n_maj = len(X_maj)
        n_min = len(X_min)

        if n_min >= n_maj:
            return X.copy(), y.copy()

        # Identify borderline (DANGER) minority instances using m nearest neighbors in all X
        m = min(self.m_neighbors, len(X) - 1)
        nn_all = NearestNeighbors(n_neighbors=m + 1, metric="euclidean", n_jobs=-1)
        nn_all.fit(X)
        all_indices = nn_all.kneighbors(X_min, return_distance=False)[:, 1:]

        # Count majority neighbors for each minority instance
        maj_counts = np.sum(y[all_indices] == 0, axis=1)

        # DANGER zone: between m/2 and m majority neighbors
        danger_mask = (maj_counts >= (m / 2.0)) & (maj_counts < m)
        danger_indices = np.where(danger_mask)[0]

        if len(danger_indices) == 0:
            danger_indices = np.arange(n_min)

        if amounts is not None:
            amounts_min = amounts[y == 1]
            costs_danger = np.maximum(amounts_min[danger_indices], 0.01)
        else:
            costs_danger = np.ones(len(danger_indices), dtype=float)

        weights = costs_danger / costs_danger.sum()
        total_synthetic = n_maj - n_min
        allocations = allocate_samples(weights, total_synthetic)

        k = min(self.k_neighbors, n_min - 1)
        nn_min = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
        nn_min.fit(X_min)
        min_indices = nn_min.kneighbors(X_min[danger_indices], return_distance=False)[:, 1:]

        synthetic_samples = []
        for idx_in_danger, orig_idx in enumerate(danger_indices):
            n_samples = allocations[idx_in_danger]
            if n_samples == 0:
                continue

            neighbor_choices = rng.choice(min_indices[idx_in_danger], size=n_samples, replace=True)
            lambdas = rng.uniform(0.0, 1.0, size=(n_samples, 1))
            diffs = X_min[neighbor_choices] - X_min[orig_idx]
            syn = X_min[orig_idx] + lambdas * diffs
            synthetic_samples.append(syn)

        if len(synthetic_samples) > 0:
            X_syn = np.vstack(synthetic_samples)
            y_syn = np.ones(len(X_syn), dtype=int)
            X_resampled = np.vstack([X, X_syn])
            y_resampled = np.concatenate([y, y_syn])
        else:
            X_resampled, y_resampled = X.copy(), y.copy()

        return X_resampled, y_resampled


class CostSensitiveSMOTEENN(BaseOversampler):
    """
    Cost-Sensitive SMOTE-ENN:
    Combines Cost-Sensitive SMOTE oversampling with Edited Nearest Neighbors cleaning.
    """

    def __init__(self, random_state=42, smote_k_neighbors=5, enn_n_neighbors=3):
        super().__init__(random_state=random_state)
        self.cs_smote = CostSensitiveSMOTE(
            random_state=random_state,
            k_neighbors=smote_k_neighbors,
        )
        self.enn = EditedNearestNeighbours(
            n_neighbors=enn_n_neighbors,
            sampling_strategy="all",
            n_jobs=-1,
        )

    def fit_resample(self, X, y, amounts=None):
        X_res, y_res = self.cs_smote.fit_resample(X, y, amounts=amounts)
        return self.enn.fit_resample(X_res, y_res)


class CostSensitiveSMOTETomek(BaseOversampler):
    """
    Cost-Sensitive SMOTE-Tomek:
    Combines Cost-Sensitive SMOTE oversampling with Tomek Links cleaning.
    """

    def __init__(self, random_state=42, smote_k_neighbors=5):
        super().__init__(random_state=random_state)
        self.cs_smote = CostSensitiveSMOTE(
            random_state=random_state,
            k_neighbors=smote_k_neighbors,
        )
        self.tomek = TomekLinks(sampling_strategy="all", n_jobs=-1)

    def fit_resample(self, X, y, amounts=None):
        X_res, y_res = self.cs_smote.fit_resample(X, y, amounts=amounts)
        return self.tomek.fit_resample(X_res, y_res)
