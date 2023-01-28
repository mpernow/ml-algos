from __future__ import annotations
import copy
import numpy as np
import scipy.stats as sps
from typing import Union, Callable, Tuple, List

from mlalgos.utils.error_funcs import misclassification_error, mean_squared_error

class Node:
    def __init__(
        self,
        parent: Node=None,
        feature: int=None,
        bound: float=None,
        left: Node=None,
        right: Node=None,
        is_leaf: bool=False,
        val: Union[int, float]=None,
        err: float=None,
        N: int=None
    ):
        """
        Represents a node in the decision tree.

        Args:
            parent            (Node): Parent node.
            feature            (int): Index of the feature to split on.
            bound            (float): Value to split on.
            left              (Node): Left child node.
            right             (Node): Right child node.
            is_leaf           (bool): Whether the node is a leaf.
            val  (Union[int, float]): Predicted value of the node if it were a leaf.
            err              (float): Error incurred by prediction from this node.
            N                  (int): Number of data points passing through this node.
        """
        self.parent = parent
        self.feature = feature
        self.bound = bound
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.val = val # present even for internal nodes, since useful for pruning
        self.err = err
        self.N = N

    def __repr__(self):
        return '(feat: '+str(self.feature)+', bound: '+str(self.bound)+')'


class DecisionTree:
    def __init__(
        self,
        penalty_function: Callable,
        ave_function: Callable,
        max_depth: int=5,
        min_samples: int=3
    ):
        """
        Initialise the decision tree, setting the stopping criteria, penalty function, and averaging function.

        Args:
            penalty_function    (Callable): The error function to use.
            ave_function        (Callable): The averaging function to use. Regression trees should have mean and
                classification trees should have mode.
            max_depth      (int, optional): Maximum depth allowed before stopping. Defaults to 5.
            min_samples    (int, optional): Minimum number of samples allowed in a node before stopping. Defaults to 3.
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.penalty_function = penalty_function
        self.ave_function = ave_function
        self.nodes = []
        self.leaves = []
    
    def split_inds(
        self,
        X: np.array,
        feature: int,
        bound: float
    ) -> Tuple[List[int], List[int]]:
        """
        Performs the split by returning the indices that are to the left and the indices that are to the right.

        Args:
            X (np.array): The input data as a Numpy array. Shape (n_samples, n_features)
            feature (int): The index of the feature to split on
            bound (float): The value of the chosen feature to split on

        Returns:
            Tuple[List[int], List[int]]: Lists of indices in the left side and the right side
        """
        inds_left = np.where((X <= bound)[:,feature])[0]
        inds_right = np.where((X > bound)[:,feature])[0]
        return inds_left, inds_right

    def best_split(
        self,
        X: np.array,
        y: np.array
    ) -> Tuple[int, float]:
        """
        Computes the best split that minimises the error incurred

        Args:
            X (np.array): Input features. Shape (n_samples, n_features)
            y (np.array): Target values. Shape (n_samples)

        Returns:
            Tuple[int, float]: The feature index and its value to split on
        """
        p = X.shape[1]
        best_ind_bound = ()
        best_err = np.inf
        for i in range(p):
            for val in np.unique(X[:,i]):
                ind_left, ind_right = self.split_inds(X, i, val)
                y_left = y[ind_left]
                y_right = y[ind_right]
                if len(y_right) == 0:
                    pred_y = np.ones(y_left.shape[0])*self.ave_function(y_left)
                    err = self.penalty_function(y, pred_y)
                elif len(y_left) == 0:
                    pred_y = np.ones(y_right.shape[0])*self.ave_function(y_right)
                    err = self.penalty_function(y, pred_y)
                else:
                    pred_y_left = np.ones(y_left.shape[0])*self.ave_function(y_left)
                    pred_y_right = np.ones(y_right.shape[0])*self.ave_function(y_right)
                    err = (y_left.shape[0]/y.shape[0])*self.penalty_function(y_left, pred_y_left) + (y_right.shape[0]/y.shape[0])*self.penalty_function(y_right, pred_y_right)
                if (err < best_err) and (y_left.shape[0] >= self.min_samples) and (y_right.shape[0] >= self.min_samples):
                    best_err = err
                    best_ind_bound = (i, val)
        return best_ind_bound

    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Calls the function to build the tree

        Args:
            X (np.array): Input features. Shape (n_samples, n_features)
            y (np.array): Target values. Shape (n_samples)
        """
        # Reset the nodes and leaves
        self.nodes = []
        self.leaves = []
        self.parent_node = self.build_tree(X, y, 0)
        self.nodes.append(self.parent_node)


    def build_tree(
        self,
        X: np.array,
        y: np.array,
        depth: int,
        parent: Node=None
    ) -> Node:
        """
        Builds the decision tree recursively

        Args:
            X (np.array): Input features. Shape (n_features, n_samples)
            y (np.array): Target values. Shape (n_features)
            depth (int): Current depth of the tree
            parent (Node, optional): The parent node from which to grow the tree. Defaults to None.

        Returns:
            Node: The parent node of the tree, containing children recursively
        """
        n_samples = y.shape[0]
        n_unique_labels = len(np.unique(y))

        val = self.ave_function(y).item() # convert from numpy.float64 to float
        
        # The following three lines are just for use in pruning
        N = y.shape[0]
        pred_y = np.ones(N)*val
        err = self.penalty_function(y, pred_y)

        if (depth >= self.max_depth or n_unique_labels == 1 or n_samples <= self.min_samples*2):
            # Stop the recursion
            node = Node(parent=parent, is_leaf=True, val=val, N=N, err=err)
            self.leaves.append(node)
            return node
        
        feature, bound = self.best_split(X, y)
        inds_left, inds_right = self.split_inds(X, feature, bound)

        node = Node(parent=parent, feature=feature, bound=bound, val=val, err=err, N=N)

        left = self.build_tree(X[inds_left, :], y[inds_left], depth+1, node)
        right = self.build_tree(X[inds_right, :], y[inds_right], depth+1, node)
        if not left.is_leaf:
            self.nodes.append(left)
        if not right.is_leaf:
            self.nodes.append(right)

        node.left = left
        node.right = right

        return node

    def traverse(
        self,
        x: np.array
    ) -> Union[int, float]:
        """
        Traverses the tree for a single input sample to predict from

        Args:
            x (np.array): Input sample. Shape (n_features)

        Returns:
            Union[int, float]: Predicted value as float or int for regression or classification, respectively
        """
        current = self.parent_node
        while not current.is_leaf:
            if x[current.feature] <= current.bound:
                current = current.left
            else:
                current = current.right
        return current.val

    def get_number_terminal(
        self
    ) -> int:
        """
        Returns the number of terminal nodes in the tree.

        Returns:
            int: Number of terminal nodes
        """
        return len(self.leaves)

    def predict(
        self,
        X: np.array
    ) -> list:
        """
        Computes the prediction for a set of input vectors.

        Args:
            X (np.array): Input data points. Shape (n_samples, n_features)

        Returns:
            list: Predictions for each data point
        """
        y = [self.traverse(x) for x in X]
        return y

    def reset(
        self
    ):
        """
        Resets the nodes and leaves of the tree.
        """
        self.nodes = []
        self.leaves = []


class RegressionTree(DecisionTree):
    def __init__(
        self,
        penalty_function: Callable,
        max_depth: int=5,
        min_samples: int=3
    ):
        """
        Initialise a regression tree, by creating a decision tree with the mean as the averaging function.

        Args:
            penalty_function    (Callable): The penalty function to use
            max_depth      (int, optional): Maximum depth of the tree. Defaults to 5.
            min_samples    (int, optional): Minimum number of training samples to allow per node. Defaults to 3.
        """
        super().__init__(penalty_function, np.mean, max_depth, min_samples)


class ClassificationTree(DecisionTree):
    def __init__(
        self,
        penalty_function: Callable,
        max_depth: int=5,
        min_samples: int=3,
        mode_function: Callable=None
    ):
        """
        Initialises a classification tree by creating a decision tree with the mode as the averaging function.

        Args:
            penalty_function (Callable): The penalty function to use
            max_depth (int, optional): Maximum allowed depth of the tree. Defaults to 5.
            min_samples (int, optional): Minimum number of training samples to allow per node. Defaults to 3.
            mode_function (Callable, optional): Function to use for the mode. If None is supplied, use self.mode.
        """
        if mode_function is None:
            mode_function = self.mode

        super().__init__(penalty_function, mode_function, max_depth, min_samples)

    def mode(
        y: np.array
    ) -> int:
        """
        Computes the mode of the input array.

        Args:
            y (np.array): Input array to find mode for. Shape (n_samples)

        Returns:
            int: The mode
        """
        return sps.mode(y).mode[0]


class Pruner:
    """Prunes a decision tree using cost-complexity pruning"""
    def __init__(
        self,
        tree: DecisionTree,
        alpha: float
    ):
        self.tree = tree
        self.alpha = alpha

    def prune_candidates(
        self,
        tree: DecisionTree
    ) -> List[Node]:
        """
        Returns list of possible nodes to prune. That is, second last layer
        
        Args:
            tree    (DecisionTree): The tree under consideration
        
        Returns
            List[Node]: A list of the nodes that can be pruned, i.e. second-to-last layer
        """
        candidates = []
        for leaf in tree.leaves:
            if leaf.parent not in candidates:
                candidates.append(leaf.parent)
        return candidates

    def prune_one(
        self,
        tree: DecisionTree,
        candidates: List[Node]
    ) -> DecisionTree:
        """
        Collapses the node that leads to the smallest per-node increase in the total error.

        Args:
            tree      (DecisionTree): The tree under consideration
            candidates  (List[Node]): List of nodes in the tree that can be pruned
        
        Returns:
            DecisionTree: The tree with one node pruned
        """
        # Check which candidate node carries least total error
        to_collapse = candidates[0]
        for candidate in candidates:
            if candidate.err < to_collapse.err:
                to_collapse = candidate
        # Collapse best candidate node
        # Need to find all the leaves that need to be removed by traversing
        leaves_to_remove = []
        nodes_to_remove = [to_collapse]
        to_check = [to_collapse]
        while any([not n.is_leaf for n in to_check]):
            if to_check[0].left.is_leaf:
                leaves_to_remove.append(to_check[0].left)
            else:
                to_check.append(to_check[0].left)
                nodes_to_remove.append(to_check[0].left)
            if to_check[0].right.is_leaf:
                leaves_to_remove.append(to_check[0].right)
            else:
                to_check.append(to_check[0].right)
                nodes_to_remove.append(to_check[0].right)
            to_check.remove(to_check[0])
        for l in leaves_to_remove:
            tree.leaves.remove(l)
        for n in nodes_to_remove:
            tree.nodes.remove(n)
        # Will be a new leaf
        tree.leaves.append(to_collapse)
        to_collapse.is_leaf = True
        # No more needed, since children not accessed if it is a leaf,
        # and the val and err have been assigned during construction of tree
        return tree

    def pruned_trees(
        self
    ) -> List[DecisionTree]:
        """
        Creates list of pruned trees by greedily pruning one node at a time.

        Returns:
            List[DecisionTree]: List of successively more pruned trees
        """
        trees = [self.tree]
        while len(trees[-1].leaves) > 2:
            # Tree has more than two leaves: keep pruning
            to_prune = copy.deepcopy(trees[-1])
            candidates = self.prune_candidates(to_prune)
            trees.append(self.prune_one(to_prune, candidates))
        return trees
    
    def prune(
        self
    ) -> DecisionTree:
        """
        Selects the best pruned tree according to cost-complexity criterion
        
        Returns:
            DecisionTree: The selected pruned tree
        """
        pruned_list = self.pruned_trees()
        cost_complexities = []
        for tree in pruned_list:
            tree_size = len(tree.leaves)
            tot_err = sum([tree.leaves[i].N * tree.leaves[i].err for i in range(len(tree.leaves))])
            cost_complexities.append(tot_err + self.alpha * tree_size)
        best_idx = cost_complexities.index(min(cost_complexities))
        return pruned_list[best_idx]


class ClassificationTreePruned:
    def __init__(
        self,
        alpha: float=0.5,
        err_func: Callable=misclassification_error,
        max_depth: int=10,
        min_samples: int=3
    ):
        """
        Initialises a pruned classification tree.

        Args:
            alpha           (float): The cost-complexity parameter. Defaults to 0.5.
            err_func     (Callable): The error function to use when building the tree. Defaults to misclassification_error.
            max_depth         (int): Maximum depth of the tree. Defaults to 10.
            min_samples       (int): Minimum number of data points per node. Defaults to 3.
        """
        self.alpha = alpha
        self.model = ClassificationTree(err_func, max_depth, min_samples)
    
    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Fits the pruned model.

        Args:
            X (np.array): Input features. Shape (n_samples, n_features)
            y (np.array): Target values. Shape (n_samples)
        """
        self.model.fit(X, y)
        pruner = Pruner(self.model, self.alpha)
        dt_classify_pruned = pruner.prune()
        self.model = dt_classify_pruned
    
    def predict(
        self,
        X: np.array
    ) -> np.array:
        """
        Prediction for a set of input features.

        Args:
            X (np.array): Input features to derive predictions for. Shape (n_points)

        Returns:
            np.array: The predicted values
        """
        return self.model.predict(X)
    
    def get_number_terminal(
        self
    ) -> int:
        """
        Returns the number of terminal nodes in the tree.

        Returns:
            int: Number of terminal nodes.
        """
        return self.model.get_number_terminal()
    
    def get_nodes(
        self
    ) -> List[Node]:
        """
        Returns the list of nodes in the tree.

        Returns:
            List[Node]: List of nodes
        """
        return self.model.nodes
    
    def reset(
        self
    ):
        """
        Resets the tree by clearing all nodes.
        """
        self.model.reset()


class RegressionTreePruned:
    def __init__(
        self,
        alpha: int=0.5,
        err_func: Callable=mean_squared_error,
        max_depth: int=10,
        min_samples: int=3
    ):
        """
        Initialises a pruned regression tree.

        Args:
            alpha             (int): The cost-complexity parameter. Defaults to 0.5.
            err_func     (Callable): Error function to use when building the tree. Defaults to mean_squared_error.
            max_depth         (int): Maximum depth of the tree. Defaults to 10.
            min_samples       (int): Minimum number of data points per node. Defaults to 3.
        """
        self.alpha = alpha
        self.model = RegressionTree(err_func, max_depth, min_samples)
    
    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Fits the pruned model.

        Args:
            X (np.array): Input features. Shape (n_samples, n_features)
            y (np.array): Target values. Shape (n_samples)
        """
        self.model.fit(X, y)
        pruner = Pruner(self.model, self.alpha)
        dt_regression_pruned = pruner.prune()
        self.model = dt_regression_pruned
    
    def predict(
        self,
        X: np.array
    ) -> np.array:
        """
        Prediction for a set of input features.

        Args:
            X (np.array): Input features to derive predictions for. Shape (n_points)

        Returns:
            np.array: The predicted values
        """
        return self.model.predict(X)
    
    def reset(
        self
    ):
        """
        Resets the tree by clearing all nodes.
        """
        self.model.reset()
