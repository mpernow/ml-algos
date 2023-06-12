import numpy as np

class LogisticRegression:
    def __init__(
        self
    ):
        """
        Initialises a logistic regression classifier.
        """
        self.n_classes = 0

    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Fits the logistic regression model by performing Newton-Raphson optimisation of the parameters.
        The target classes are labelled 0, 1, ... and each one is expected to appear in y at least once.

        Args:
            X (np.array): The features as an array of shape (n_samples, n_features)
            y (np.array): The targets as an array of shape (n_samples)
        """
        n_classes = len(np.unique(y))
        N = y.shape[0]

        X_with_bias = np.insert(X, 0, 1, axis=1)

        betas = np.zeros((X_with_bias.shape[1], n_classes - 1))

        stacked_y = self._get_stacked_y(y)
        # TODO
        # Set initial betas
        # Compute derivative and Hessian
        # Optimise

        self.n_classes = n_classes

    @staticmethod
    def _get_stacked_y(
        y: np.array,
        n_classes: int,
        N: int
    ) -> np.array:
        """
        Creates a stacked one-hot encoding of y.
        The result is [y1==0, y2==0, ..., yN==0, y1==1, ..., yN==1, ..., y1==K-1, ..., yN==K-1]

        Args:
            y    (np.array): The original array of classes for each data point
            n_classes (int): The number of classes
            N         (int): The number of data points
        """
        stacked_y = np.zeros((n_classes * N))
        idx_ones = N * y + np.arange(N)
        stacked_y[idx_ones] = 1
        return stacked_y
