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
        stacked_p = self._get_stacked_probabilites(X, betas, n_classes)
        # TODO
        # Set initial betas
        # Compute derivative and Hessian
        # Optimise

        self.n_classes = n_classes

    def _get_stacked_y(
        self,
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

    def _get_stacked_probabilites(
        self,
        X: np.array,
        betas: np.array,
        n_classes: int
    ) -> np.array:
        """
        Computes the stacked vector of logit probabilities.
        The result is [p_0(x_1), p_0(x_2), ..., p_0(x_N), p_1(x_1), ..., p_{K-2}(x_N)]
        which includes only the first K-1 classes (0 up to K-2) since the last class has no independent beta parameters.

        Args:
            X     (np.array): The data features as an array of shape (n_samples, n_features)
            betas (np.array): The beta parameters of the logistic regression model as an array of shape (n_features, n_classes - 1)
            n_classes  (int): The number of classes
        """
        beta_times_X = betas.T @ X.T # shape (n_classes - 1, n_samples)
        denominators = 1 + np.exp(np.sum(beta_times_X, axis=1)) # shape (1, n_samples)
        logits = np.exp(beta_times_X) / np.tile(denominators, (n_classes - 1, 1)) # shape (n_classes - 1, n_samples)
        return logits.T.flatten()
