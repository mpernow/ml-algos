import numpy as np


class StandardScaler:
    def __init__(
        self
    ):
        """
        Initialises the mean and standard deviation parameters to 0 and 1 respectively.
        """
        self.mean = 1.
        self.std = 0.

    def fit(
        self,
        X_train: np.array
    ):
        """
        Computes the mean and standard deviation of the dataset X_train.

        Args:
            X_train   (np.array): The data to compute the mean and std from. Shape (n_samples, n_features)
        """
        self.mean = np.mean(X_train, axis=0, keepdims=True)
        self.std = np.std(X_train, axis=0, keepdims=True)

    def transform(
        self,
        X: np.array
    ) -> np.array:
        """
        Computes the transformation z = (X - mean)/std.

        Args:
            X   (np.array): The dataset to standardise. Shape (n_samples, n_features)
        
        Returns:
            np.array:   The transformed data
        """
        return (X - self.mean) / self.std
