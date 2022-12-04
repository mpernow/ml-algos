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


class RidgeRegression:
    def __init__(
        self,
        lam=1
    ):
        """
        Initialises the beta_hat parameter to None and the regularisation parameter to specified.

        Args:
            lam (float, default=1): The regularisation parameter
        """
        self.lam = lam
        self.beta_hat = None
    
    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Computes the optimal parameters beta_hat for the given training data using ridge regression.
        Note that the input ought to be standardised, since ridge regression is not invariant under scaling.

        Args:
            X   (np.array): Training data of shape (n_samples, n_features)
            y   (np.array): Target values for the training data with shape (n_samples)
        """
        # Create a penalty term
        p = X.shape[1] + 1
        penalty_term = self.lam * np.eye(p)
        # Set the (0,0) element to zero so as not to penalise intercept parameter
        penalty_term[0][0] = 0
        # Add a feature row of ones for the intercept
        X_with_bias = np.insert(X, 0, 1., axis=1)
        # Compute the coefficients
        self.beta_hat = spl.inv(X_with_bias.T @ X_with_bias + penalty_term) @ X_with_bias.T @ y
    
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
        return X @ self.beta_hat[1:] + self.beta_hat


class LassoRegression:
    def __init__(
        self,
        lam=1
    ):
        """
        Initialises the beta_hat parameter to None and the regularisation parameter to the specified value.

        Args:
            lam (float, default=1): The regularisation parameter
        """
        self.lam = lam
        self.beta_hat = None
    
    def fit(
        self,
        X: np.array,
        y: np.array,
        num_epochs: int=500,
        learning_rate: float=0.001
    ):
        """
        Computes the optimal parameters beta_hat for the given training data using lasso regression.
        Note that the input ought to be standardised, since ridge regression is not invariant under scaling.

        Args:
            X            (np.array): Training data of shape (n_samples, n_features)
            y            (np.array): Target values for the training data with shape (n_samples)
            num_epochs        (int): Number of epochs to train for
            learning_rate   (float): Learning rate during training
        """
        b = np.random.uniform(-1., 1., (X.shape[1] + 1, ))
        for _ in range(num_epochs):
            y_hat = X @ b[1:] + b[0]
            d = y_hat - y
            reg_term = np.sign(b)
            reg_term[0] = 0
            b = b - learning_rate*(np.insert(X,0,1,axis=1).T @ d + self.lam * reg_term)
        self.beta_hat = b
    
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
        return X @ self.beta_hat[1:] + self.beta_hat
