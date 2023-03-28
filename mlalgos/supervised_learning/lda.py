import numpy as np
import scipy.linalg as spl


class LDA:
    def __init__(
        self
    ):
        """
        Initialises a linear discriminant analysis classifier. The classes are assumed to be labelled 0, 1, ...
        """
        self.n_classes = 0
        self.covariance = None
        self.class_centroids = []
        self.class_priors = []
    
    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Fits the LDA classifier by computing the centroids, priors, and covariance.
        The target classes are expected to be 0, 1, ... and each of them have to appear
        in y at least once

        Args:
            X (np.array): The features as an array of shape (n_samples, n_features)
            y (np.array): The targets as an array of shape (n_samples)
        """
        n_classes = len(np.unique(y))
        N = len(y)
        
        # Compute centroids and priors
        centroids = []
        priors = []
        for cl in range(n_classes):
            centroids.append(np.mean(X[y == cl], axis=0))
            priors.append(float(np.sum(y == cl))/N)
        
        # Compute variance (pooled, unbiased)
        var = np.zeros((X.shape[1], X.shape[1]))
        for cl, mu in zip(range(n_classes), centroids):
            class_var = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == cl]:
                row, mu = row.reshape(X.shape[1], 1), mu.reshape(X.shape[1], 1)
                class_var += (row - mu) @ (row - mu).T
            var += class_var
        var /= (N - n_classes)

        self.n_classes = n_classes
        self.class_centroids = centroids
        self.class_priors = priors
        self.covariance = var

    def predict(
        self,
        X: np.array
    ) -> np.array:
        """
        Uses the computed parameters in fitting to return the class with highest discriminant function.

        Args:
            X (np.array): Features to predict from as an array of shape (n_samples, n_features)
        
        Returns:
            np.array: The predicted values as an array of shape (n_samples)
        """
        discriminators = []
        inv_sig = spl.inv(self.covariance)
        for cl in range(self.n_classes):
            discriminators.append(X @ inv_sig @ self.class_centroids[cl]
                - 0.5 * self.class_centroids[cl].T @ inv_sig @ self.class_centroids[cl]
                + np.log(self.class_priors[cl]))
        return np.argmax(discriminators, axis=0)

    def transform(self,
        X: np.array,
        n_components: int
    ) -> np.array:
        """
        Uses the computed parameters to perform an LDA rank reduction. The notation is that of ESL Chapter 4.3

        Args:
            X       (np.array): Input features to transform as an array of shape (n_samples, n_features)
            n_components (int): Number of components to keep

        Returns:
            np.array: Transformed array of shape (n_samples, n_components)
        """
        M = np.array(self.class_centroids)
        W = self.covariance  # within-class covariance
        W_inv_sqrt = spl.inv(spl.fractional_matrix_power(W, 0.5))
        M_star = M @ W_inv_sqrt

        # Compute between-class covariance
        mean_of_means = np.zeros((M_star.shape[1],))
        for cl in range(self.n_classes):
            mean_of_means += self.class_priors[cl] * M_star[cl]
        mean_of_means = mean_of_means.reshape(M_star.shape[1], 1)
        B_star = np.zeros((M_star.shape[1], M_star.shape[1]))
        for cl in range(self.n_classes):
            M_k = M_star[cl].reshape(M_star.shape[1], 1)
            B_star += self.class_priors[cl] * (M_k - mean_of_means) @ (M_k - mean_of_means).T
        
        DB, V_star = spl.eigh(B_star)
        V = W_inv_sqrt @ V_star

        # The columns of V are the eigenvectors. Now form v_l^T X
        Z = X @ V[:, :n_components]

        return Z 
