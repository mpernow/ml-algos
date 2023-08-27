import numpy as np

class LogisticRegression:
    def __init__(
        self
    ):
        """
        Initialises a logistic regression classifier.
        """
        self.n_classes = 0
        self.betas = np.array([])

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

        betas = np.zeros(( X_with_bias.shape[1], (n_classes - 1), )) + 0.01 # Not exactly zero to avoid singular matrix in Newton step
        # betas = np.random.rand(X_with_bias.shape[1], (n_classes - 1))

        tol = 1.e-3
        diff = 1.
        diffs = []
        losses = []
        # while diff > tol:
        for i in range(20):
            print(betas)
            logits = self._get_logits(X_with_bias, betas, n_classes)
            print(logits)
            print("loss: ", self.loss(y, logits))
            losses.append(self.loss(y, logits))
            first_deriv = self._first_deriv(X_with_bias, y, betas, n_classes, N)
            second_deriv = self._second_deriv(X_with_bias, betas, n_classes, N)
            new_betas = self._newton_step(betas, first_deriv, second_deriv)
            diff = np.linalg.norm(new_betas - betas)
            print('New:')
            print(new_betas)
            print('Old:')
            print(betas)
            print(diff)
            diffs.append(diff)
            print()
            betas = new_betas

        self.n_classes = n_classes
        self.betas = betas
        print(losses)
        print(diffs)

    def loss(self, y, logits):
        from sklearn.metrics import log_loss
        return log_loss(y, logits)

    def predict(
        self,
        X: np.array
    ) -> np.array:
        """
        Computes a prediction of the logistic regression model for the given input parameters.

        Args:
            X (np.array): The input data of shape (n_samples, n_features)

        Returns:
            np.array: Predicted values as an np.array of shape (n_samples)
        """
        X_with_bias = np.insert(X, 0, 1, axis=1)
        logits = self._get_logits(X_with_bias, self.betas, self.n_classes)
        return np.argmax(logits, axis=1)

    def _newton_step(
        self,
        betas: np.array,
        first_deriv: np.array,
        second_deriv: np.array
    ) -> np.array:
        """
        Computes the next beta in a Newton-Raphson step.

        Args:
            betas        (np.array): The beta parameters as an array of shape (n_features * (n_classes - 1), )
            first_deriv  (np.array): First derivative as an np.array
            second_deriv (np.array): Second derivative as an np.array
        """
        return betas - (np.linalg.pinv(second_deriv) @ first_deriv).reshape(betas.shape, order='F')

    def _first_deriv(
        self,
        X: np.array,
        y: np.array,
        betas: np.array,
        n_classes: int,
        N: int
    ) -> np.array:
        """
        Computes the first derivatives of the log-likelihood with respect to the parameters beta.

        Args:
            X     (np.array): The features as an array of shape (n_samples, n_features)
            y     (np.array): The targets as an array of shape (n_samples)
            betas (np.array): The beta parameters as an array of shape (n_features * (n_classes - 1), )
            n_classes  (int): The number of classes
            N          (int): The number of data points
        """
        stacked_y = self._get_stacked_y(y, n_classes, N)
        stacked_p = self._get_stacked_probabilites(X, betas, n_classes)
        X_hat = self._get_block_x(X, n_classes)
        return X_hat @ (stacked_y - stacked_p) # shape n_features * (n_classes - 1)

    def _second_deriv(
        self,
        X: np.array,
        betas: np.array,
        n_classes: int,
        N: int
    ) -> np.array:
        """
        Computes the second derivative (Hessian) of the log-likelihood with respect to the parameters beta.

        Args:
            X     (np.array): The features as an array of shape (n_samples, n_features)
            betas (np.array): The beta parameters as an array of shape (n_features * (n_classes - 1), )
            n_classes  (int): The number of classes
            N          (int): The number of data points
        """
        stacked_p = self._get_stacked_probabilites(X, betas, n_classes)
        p_mats = [np.diag(stacked_p[k * N: (k + 1) * N]) for k in range(n_classes - 1)]
        w_diags = [-p @ (np.eye(N) - p) for p in p_mats]
        w_mats = [[p_mats[i] @ p_mats[j] for j in range(n_classes - 1)] for i in range(n_classes - 1)]
        for i in range(n_classes - 1):
            w_mats[i][i] = w_diags[i]
        W = np.block(w_mats)
        X_hat = self._get_block_x(X, n_classes)
        return X_hat @ W @ X_hat.T

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
        return stacked_y[:N * (n_classes - 1)]

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
            betas (np.array): The beta parameters of the logistic regression model as an array of shape (n_features * (n_classes - 1), )
            n_classes  (int): The number of classes
        
        Returns:
            np.array: The probabilities as a stacked array of shape (n_samples * (n_classes - 1))
        """
        logits = self._get_logits(X, betas, n_classes)
        # Remove last class (since it has no independent parameters)
        return logits[:, :-1].T.flatten()

    def _get_block_x(
        self,
        X: np.array,
        n_classes: int
    ):
        """
        Creates matrix of block diagonal duplicates of X.

        Args:
            X     (np.array): The data features as an array of shape (n_samples, n_features)
            n_classes  (int): The number of classes
        """
        return np.kron(np.eye(n_classes - 1), X.T)
    
    def _get_logits(
        self,
        X: np.array,
        betas: np.array,
        n_classes: int
    ) -> np.array:
        """
        Computes the logit probabilites for a given input.

        Args:
            X     (np.array): The data features as an array of shape (n_samples, n_features)
            betas (np.array): The beta parameters of the logistic regression model as an array of shape (n_features * (n_classes - 1), )
            n_classes  (int): The number of classes

        Returns:
            np.array: The logit probabilities as an array of shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        beta_times_X = X @ betas # shape (n_samples, n_classes - 1)
        denominators = 1 + np.exp(np.sum(beta_times_X, axis=1, keepdims=True)) # shape (n_samples, 1)
        logits = np.exp(beta_times_X) / np.tile(denominators, (1, n_classes - 1)) # shape (n_samples, n_classes - 1)
        # Add in the last column (corresponding to last class)
        return np.concatenate((logits, (1 - np.sum(logits, axis=1)).reshape((n_samples, 1))), axis=1)