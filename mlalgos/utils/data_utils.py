import numpy as np
import torch


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


def compute_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """
    Computes the accuracy of a trained pytorch model.

    Args:
        model                   (torch.nn.Module): The model to predict using
        data_loader (torch.utils.data.Dataloader): Dataloader contianing the data
        device                     (torch.device): Pytorch device to run model on

    Returns:
        float: Computed accuracy as a percentage
    """
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            probs = model(features)
            # If split over multiple devices
            if isinstance(probs, torch.distributed.rpc.api.RRef):
                probs = probs.local_value()
            _, predicted_labels = torch.max(probs, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum().float()
    return correct_pred/num_examples * 100


class TransformPCA:
    """
    Implement the PCA transformation of the AlexNet paper.
    """
    def __init__(self):
        pass

    def __call__(self, image):
        # Make sure image is e.g. (3, 32, 32)
        (h, w) = (image.shape[1], image.shape[2])
        assert (image.dim() == 3) and (image.shape[0] == 3)

        # Normalise to zero mean and unit std
        mean = image.mean((1,2), keepdim=True)
        std = image.std((1,2), keepdim=True)
        normalised_image = (image - mean)/std

        flattened = normalised_image.flatten(start_dim=1, end_dim=2)
        # Make sure it is (3, d)
        assert (flattened.dim() == 2) and (flattened.shape[0] == 3)
        
        # Compute covariance where rows are variables
        cov = flattened.cov()
        # Should be 3x3
        assert (cov.shape == (3,3))

        # Eigendecomposition (eigenvectors are columns of v)
        # Use eigh since covariance is real symmetric, giving real eigen-decomposition
        l, p = torch.linalg.eigh(cov)

        # Random variables
        alphas = torch.normal(mean=0.0, std=0.1, size=(3,))
        # alphas = torch.normal(mean=0.0, std=0.1, size=(3,))
        
        # Define the perturbations as the three principal components scaled by eigenvector times random variable
        # (3x3) x (3x1) --> (3x1)
        delta = p @ (alphas * l).reshape((3, 1))
        perturbed = (flattened + delta).reshape((3, h, w))

        # Clear some variables, to prevent slowing down with epochs (not sure is this made any difference)
        del h
        del w
        del normalised_image
        del flattened
        del cov
        del l
        del p
        del alphas
        del delta

        return (perturbed * std + mean)