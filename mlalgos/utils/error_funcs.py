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


def misclassification_error(
    y_true: np.array,
    y_pred: np.array
) -> float:
    """
    Computes the misclassification error between the true and predicted values.

    Args:
        y_true (np.array): The true values as a numpy array of shape (n_pred)
        y_pred (np.array): The predicted values as a numpy array of shape (n_pred)

    Returns:
        float: The fraction of missclassified values
    """
    N = len(y_true)
    return (1./N) * np.sum(np.array(y_true) != np.array(y_pred))


def gini_index(
    y_true: np.array,
    y_pred: np.array
) -> float:
    """
    Computes the gini impurity index for a set of values

    Args:
        y_true (np.array): The true values as a numpy array of shape (n_pred)
        y_pred (np.array): The predicted values as a numpy array of shape (n_pred).
            Not used but included for compatibility.

    Returns:
        float: The gini index
    """
    # y_pred is only included for compatibility, it will always be mode(y_true)
    N = len(y_true)
    sum = 0
    for val in np.unique(y_true):
        p = (1./N) * np.sum(y_true == val)
        sum += p * (1 - p)
    return sum


def cross_entropy(
    y_true: np.array,
    y_pred: np.array
) -> float:
    """
    Computes the cross entropy for a set of values

    Args:
        y_true (np.array): The true values as a numpy array of shape (n_pred)
        y_pred (np.array): The predicted values as a numpy array of shape (n_pred).

    Returns:
        float: The cross entropy
    """
    N = len(y_true)
    sum = 0
    for val in np.unique(y_true):
        indxs = np.where(y_true == val)[0]
        sum -= np.sum(np.log([y_pred[indxs][:, val]]))
    return sum/N