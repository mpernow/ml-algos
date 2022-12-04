import numpy as np
import scipy.linalg as spl

class PCA:
    def __init__(
        self,
        n_components: int
    ):
        """
        Initialises the number of components to return.

        Args:
            n_components    (int): Number of principal components to return.
        """
        self.n_components = n_components
    
    def fit(
        self,
        X: np.array
    ) -> np.array:
        """
        Performs the PCA of the input data. The data should be centred on the origin before calling this function.

        Args:
            X   (np.array): The input data as an array of shape (N, p) where N is the number of
                data points and p is the number of features.
        
        Returns:
            np.array: Array containing the n_components principal components. Shape (N, n_components)
        """
        u, s, vh = spl.svd((X), full_matrices=True)
        s_mat = spl.diagsvd(s, X.shape[0], X.shape[1])
        X_transformed = u[:, 0:self.n_components] @ s_mat[0:self.n_components, 0:self.n_components]
        return X_transformed