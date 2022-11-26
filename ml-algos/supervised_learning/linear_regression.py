import numpy as np
import scipy.linalg as spl

class LinearRegression:
    def __init__(
        self
    ):
        self.beta_hat = None

    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Compute the optimal parameters given training data.

        Args:
            X   (np.array): Training data of shape (n_samples, n_features)
            y   (np.array): Target values for the training data with shape (n_samples)
        """
        # Add a feature row of ones for the intercept
        X_with_bias = np.insert(X, 0, 1, axis=1)
        # Compute the coefficients
        self.beta_hat = spl.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    
    def predict(
        self,
        X: np.array
    ):
        """
        Predicts the output of the model for given input features.

        Args: 
            X   (np.array): Input data of shape (n_samples, n_features)
        
        Returns: np.array with the output for the given input data. Shape (n_samples)
        """
        return X @ self.beta_hat[1:] + self.beta_hat[0]