import numpy as np
import scipy.linalg as spl

from mlalgos.unsupervised_learning.pca import PCA


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


class DummyRegression:
    def __init__(
        self
    ):
        self.mean = None

    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Compute the mean of the training data (only the y-values) for dummy regression.
        The X array is included as an argument for compatibility.

        Args:
            X   (np.array): Training data of shape (n_samples, n_features)
            y   (np.array): Target values for the training data with shape (n_samples)
        """
        self.mean = np.mean(y)
    
    def predict(
        self,
        X: np.array
    ):
        """
        Predicts the output of the model for given input features.
        The X array is only used for determining the shape of the output array.

        Args: 
            X   (np.array): Input data of shape (n_samples, n_features)
        
        Returns: np.array with the output for the given input data. Shape (n_samples)
        """

        return self.mean * np.ones(X.shape[0])


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
        return X @ self.beta_hat[1:] + self.beta_hat[0]


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

        Args:
            X            (np.array): Training data of shape (n_samples, n_features)
            y            (np.array): Target values for the training data with shape (n_samples)
            num_epochs        (int): Number of epochs to train for
            learning_rate   (float): Learning rate during training
        """
        # Initialise parameters
        b = np.random.uniform(-1., 1., (X.shape[1] + 1, ))
        # Train using gradient descent
        for _ in range(num_epochs):
            y_hat = X @ b[1:] + b[0]
            d = y_hat - y
            # Regularise with L1 norm, set the regularisation of the intercept to zero
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
        return X @ self.beta_hat[1:] + self.beta_hat[0]


class PrincipalComponentsRegression:
    def __init__(
        self,
        n_components: int
    ):
        """
        Initialises the beta_hat parameter to None and the number of principal components to the input value.
        """
        self.beta_hat = None
        self.n_components = n_components
    
    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Computes the optimal parameters beta_hat for the specified number of principal components.
        As when performing PCA, the input parameters are assumed to be centred.

        Args:
            X   (np.array): Training data of shape (n_samples, n_features)
            y   (np.array): Target values for the training data with shape (n_samples)
        """
        pca = PCA(self.n_components)
        Z = pca.fit(X)

        beta_hat0 = np.mean(y)
        y = y - beta_hat0
        theta_hat = np.array([(Z[:,i] @ y) / (Z[:,i] @ Z[:,i]) for i in range(self.n_components)])

        # Since we previously only had Z = X @ V, we need to form the V to use in predictions. Do this by pseudo-inverse
        V = spl.pinv(X) @ Z
        beta_hat = np.sum(np.array([theta_hat[i] @ V[:,i] for i in range(self.n_components)]), axis=0)
        self.beta_hat = np.insert(beta_hat, 0, beta_hat0)
    
    def predict(
        self,
        X: np.array
    ):
        """
        Predicts the output of the model for the given input features.

        Args:
            X   (np.array): Input data of shape (n_samples, n_features)

        Returns: np.array with the output for the given input data. Shape (n_samples)
        """
        return X @ self.beta_hat[1:] + self.beta_hat[0]

class PartialLeastSquaresRegression:
    def __init__(
        self,
        n_components: int
    ):
        """
        Initialises the beta_hat parameter to None and the number of components to the input value.
        """
        self.beta_hat = None
        self.n_components = n_components
    
    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Computes the optimal parameters beta_hat using the partial least squares method, using the specified number of components.

        Args:
            X   (np.array): Training data of shape (n_samples, n_features)
            y   (np.array): Target values for the training data with shape (n_samples)
        """
        ym = np.mean(y)
        y = y - np.mean(y)
        Xm = X
        p = X.shape[1]
        betam = np.zeros(p)
        factor = np.eye(p)
        for _ in range(self.n_components):
            phi = Xm.T @ y
            z = Xm @ phi
            theta = z @ y / (z @ z)
            Xm = Xm - np.outer(z, (z @ Xm))/(z @ z)
            betam = betam + factor @ phi * theta
            factor = factor @ (np.eye(p) - ((z @ Xm)/(z @ z)) @ phi)
        self.beta_hat = np.insert(betam, 0, ym)
        self.beta_hat = betam

    def predict(
        self,
        X: np.array
    ):
        """
        Predicts the output of the model for the given input features.

        Args:
            X   (np.array): Input data of shape (n_samples, n_features)

        Returns: np.array with the output for the given input data. Shape (n_samples)
        """
        return X @ self.beta_hat[1:] + self.beta_hat[0]