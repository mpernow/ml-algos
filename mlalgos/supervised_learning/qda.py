import numpy as np
import scipy.linalg as spl


class QDA:
    def __init__(
        self
    ):
        """
        Initialises a quadratic discriminant analysis classifier. The classes are assumed to be labelled 0, 1, ...
        """
        self.n_classes = 0
        self.class_covariances = []
        self.class_centroids = []
        self.class_priors = []
    
    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Fits the QDA classifier by computing the centroids, priors, and covariances.

        Args:
            X (np.array): The features as an array of shape (n_samples, n_features)
            y (np.array): The targets as an array of shape (n_samples)
        """
        n_classes = len(np.unique(y))
        N = len(y)
        
        # Compute centroids, priors, and covariances
        centroids = []
        priors = []
        covariances = []
        for cl in range(n_classes):
            centroid = np.mean(X[y == cl], axis=0)
            centroids.append(centroid)
            N_k = np.sum(y == cl)
            priors.append(float(N_k)/N)
            class_cov = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == cl]:
                row, mu = row.reshape(X.shape[1], 1), centroid.reshape(X.shape[1], 1)
                class_cov += (row - mu) @ (row - mu).T
            covariances.append(class_cov/(N_k - 1))
        
        self.n_classes = n_classes
        self.class_centroids = centroids
        self.class_priors = priors
        self.class_covariances = covariances

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
        for cl in range(self.n_classes):
            inv_sig = spl.inv(self.class_covariances[cl])
            quad_term = np.array([(x - self.class_centroids[cl]) @ inv_sig @ (x - self.class_centroids[cl]).T for x in X])
            discriminators.append(
                -0.5 * np.log(spl.det(self.class_covariances[cl]))
                -0.5 * quad_term
                + np.log(self.class_priors[cl])
            )
        return np.argmax(discriminators, axis=0)

