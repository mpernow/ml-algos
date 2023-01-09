import numpy as np


def mean_squared_error(
    y_true: np.array,
    y_pred: np.array
) -> float:
    """
    Computes the mean squared error between the true values and the predicted values.

    Args:
        y_true  (np.array): True values as numpy array of shape (n_pred)
        y_pred  (np.array): Predicted values as numpy array of shape (n_pred)
    
    Returns:
        float: The computed mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)
