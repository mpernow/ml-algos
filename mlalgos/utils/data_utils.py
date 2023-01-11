import numpy as np


class StandardScaler:
    def __init__(
        self
    ):
        """
        Initialises the mean and standard deviation parameters to 0 and 1 respectively.
        """
        self.mean = 1.
        self.std = 0.

    def fit(
        self,
        X_train: np.array
    ):
        """
        Computes the mean and standard deviation of the dataset X_train.

        Args:
            X_train   (np.array): The data to compute the mean and std from. Shape (n_samples, n_features)
        """
        self.mean = np.mean(X_train, axis=0, keepdims=True)
        self.std = np.std(X_train, axis=0, keepdims=True)

    def transform(
        self,
        X: np.array
    ) -> np.array:
        """
        Computes the transformation z = (X - mean)/std.

        Args:
            X   (np.array): The dataset to standardise. Shape (n_samples, n_features)
        
        Returns:
            np.array:   The transformed data
        """
        return (X - self.mean) / self.std


def cross_validation_split(
    X: np.array,
    y: np.array,
    k: int,
    shuffle: bool=True
) -> list[dict]:
    """
    Splits the dataset into k sets of k-fold cross-validated data.

    Args:
        X (np.array): The input features. Shape (n_samples, n_features)
        y (np.array): The target values. Shape (n_samples)
        k (int): The number of splits of the data.
        shuffle (bool, optional): Whether to shuffle the data prior to splitting. Defaults to True.

    Returns:
        list[dict]: A list containing the splits as dicts. Dicts have keys 'train' and 'test', which are each dicts with keys 'X' and 'y'
    """
    if shuffle:
        idx = np.random.permutation(X.shape[0])
        X, y =  X[idx], y[idx]

    n_samples = X.shape[0]
    residuals = {}
    n_residuals = (n_samples % k)
    if n_residuals != 0:
        residuals['X'] = X[-n_residuals:]
        residuals['y'] = y[-n_residuals:]
        X = X[:-n_residuals]
        y = y[:-n_residuals]
    else:
        # If no residuals, set it to empty arrays for compatibility later
        residuals['X'] = np.empty((1, X.shape[1]))
        residuals['y'] = np.empty((1))

    X_split = np.split(X, k)
    y_split = np.split(y, k)
    splits = []
    for i in range(k):
        X_test, y_test = X_split[i], y_split[i]
        # Concatenate the rest into the training data, with the residual samples
        X_train = np.concatenate(X_split[:i] + X_split[i + 1:] + [residuals['X']], axis=0)
        y_train = np.concatenate(y_split[:i] + y_split[i + 1:] + [residuals['y']], axis=0)
        splits.append({'train': {'X': X_train, 'y': y_train},
                        'test': {'X': X_test, 'y': y_test}})

    return splits