import numpy as np
import scipy.stats as sps
from typing import Any


class KNN:
    """
    Implements a simple k nearest neighbours algorithm.

    Args:
        k   (int): Number of neighbours that vote on the class of the point under consideration.
    """
    def __init__(
        self,
        k: int = 10
    ):
        """
        Initialises the number of nearest neighbours to use.

        Args:
            k (int): Number of neighbours that vote on the class of the point under consideration.
        """
        self.k = k
    
    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Loads the known data points to use in the nearest neighbour classification.

        Args:
            X (np.array): The input features. Shape (n_samples, n_features)
            y (np.array): The true classes of each data point. Shape (n_samples)
        """
        self.X = X
        self.y = y
    
    def vote_nearest(
        self,
        point: np.array
    ) -> Any:
        """
        Predicts the class of the current point from its nearest neighbours.

        Args:
            point (np.array): The point under consideration.

        Returns:
            Any: The class that the point belongs to
        """
        dists = np.linalg.norm(point - self.X, axis=1)
        indx = np.argpartition(dists, self.k)[:self.k]
        out = sps.mode(self.y[indx])[0][0]
        return out

    def predict(
        self,
        X: np.array
    ):
        if len(X.shape) == 1:
            return self.vote_nearest(X)
        elif len(X.shape) == 2:
            y_out = []
            for point in X:
                y_out.append(self.vote_nearest(point))
            return np.array(y_out)