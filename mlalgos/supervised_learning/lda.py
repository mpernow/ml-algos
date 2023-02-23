import numpy as np


class LDA:
    def __init__(
        self
    ):
        """
        Initialises a linear discriminant analysis classifier. The classes are assumed to be labelled 0, 1, ...
        """
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

        self.class_centroids = centroids
        self.class_priors = priors
        self.covariance = var

