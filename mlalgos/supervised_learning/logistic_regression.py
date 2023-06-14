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
        The result is [y_1==0, y_2==0, ..., y_N==0, y_1==1, ..., y_N==1, ..., y_1==K-2, ..., y_N==K-2]^T
        It only includes the first K-1 classes (0 up to K-2) since the last class has no independent beta parameters.

        Args:
            y    (np.array): The original array of classes for each data point
            n_classes (int): The number of classes
            N         (int): The number of data points
        """
        stacked_y = np.zeros((n_classes * N))
        idx_ones = N * y + np.arange(N)
        stacked_y[idx_ones] = 1
        return stacked_y[:N * (n_classes) - 1]
