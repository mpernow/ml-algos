import numpy as np


class LDA:
    def __init__(
        self
    ):
        """
        Initialises a linear discriminant analysis classifier.
        """
        self.covariance = None
        self.class_means = []
        self.class_priors = []
    
    def fit(
        X: np.array,
        y: np.array
    ):
        """
        Fits the LDA classifier.

        Args:
            X (np.array): The features as an array of shape (n_samples, n_features)
            y (np.array): The targets as an array of shape (n_samples)
        """
        pass