import numpy as np


def standardise(
    X: np.array
) -> np.array:
    """
    Standardises the dataset X by computing the standard score z = (x - mean)/std.

    Args:
        X   (np.array): The data to be standardised. Shape (n_samples, n_features)
    
    Returns:
        z   (np.array): The standard scores. Shape (n_samples, n_features)
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    return (X - mean) / std
