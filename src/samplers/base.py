from abc import ABC, abstractmethod
import numpy as np


class BaseOversampler(ABC):
    """
    Abstract Base Class for all oversampling strategies.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    @abstractmethod
    def fit_resample(self, X, y, amounts=None):
        """
        Resamples the dataset.
        Parameters:
            X: np.ndarray of features (standardized)
            y: np.ndarray of binary labels (0: majority, 1: minority)
            amounts: np.ndarray of unscaled transaction amounts (cost)
        Returns:
            X_resampled, y_resampled
        """
        pass
