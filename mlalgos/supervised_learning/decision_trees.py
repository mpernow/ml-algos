import copy
import numpy as np
import scipy.stats as sps


class Node:
    def __init__(
        self,
        parent=None,
        feature=None,
        bound=None,
        left=None,
        right=None,
        is_leaf=False,
        val=None,
        err=None,
        N=None
    ):
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
        penalty_function,
        ave_function,
        max_depth=5,
        min_samples=3
    ):
        """Sets the stopping criteria and penalty function"""
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.penalty_function = penalty_function
        self.ave_function = ave_function
        self.nodes = []
        self.leaves = []
    
    def split_inds(
        self,
        X,
        feature,
        bound
    ):
        """Returns indices of the left and right split"""
        inds_left = np.where((X <= bound)[:,feature])[0]
        inds_right = np.where((X > bound)[:,feature])[0]
        return inds_left, inds_right

    def best_split(
        self,
        X,
        y
    ):
        """Returns the best index and bound for splitting"""
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
        X,
        y
    ):
        """Calls the function to build the tree"""
        # Reset the nodes and leaves
        self.nodes = []
        self.leaves = []
        self.parent_node = self.build_tree(X, y, 0)
        self.nodes.append(self.parent_node)


    def build_tree(
        self,
        X,
        y,
        depth,
        parent=None
    ):
        """Builds the tree through recursion"""
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
        x
    ):
        """Traverses the tree given an input vector x"""
        current = self.parent_node
        while not current.is_leaf:
            if x[current.feature] <= current.bound:
                current = current.left
            else:
                current = current.right
        return current.val

    def get_number_terminal(self):
        return len(self.leaves)

    def predict(
        self,
        X
    ):
        y = [self.traverse(x) for x in X]
        return y

    def reset(self):
        self.nodes = []
        self.leaves = []

class RegressionTree(DecisionTree):
    def __init__(
        self,
        penalty_function,
        max_depth=5,
        min_samples=3
    ):
        super().__init__(penalty_function, np.mean, max_depth, min_samples)

def mode(y):
    return sps.mode(y).mode[0]

class ClassificationTree(DecisionTree):
    def __init__(
        self,
        penalty_function,
        max_depth=5,
        min_samples=3,
        mode_function=mode
    ):
        super().__init__(penalty_function, mode_function, max_depth, min_samples)


class Pruner:
    """Takes care of the pruning of a DecisionTree"""
    def __init__(
        self,
        tree,
        alpha
    ):
        self.tree = tree
        self.alpha = alpha

    def prune_candidates(
        self,
        tree
    ):
        """Returns list of possible nodes to prune. That is, second last layer"""
        candidates = []
        for leaf in tree.leaves:
            if leaf.parent not in candidates:
                candidates.append(leaf.parent)
        return candidates

    def prune_one(
        self,
        tree,
        candidates
    ):
        """Collapses one node, that leads to the smallest per-node increase in total error"""
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

    def pruned_trees(self):
        """Creates list of pruned trees to select from"""
        trees = [self.tree]
        while len(trees[-1].leaves) > 2:
            # Tree has more than two leaves: keep pruning
            to_prune = copy.deepcopy(trees[-1])
            candidates = self.prune_candidates(to_prune)
            trees.append(self.prune_one(to_prune, candidates))
        return trees
    
    def prune(self):
        """Selects the best pruned tree"""
        pruned_list = self.pruned_trees()
        cost_complexities = []
        for tree in pruned_list:
            tree_size = len(tree.leaves)
            tot_err = sum([tree.leaves[i].N * tree.leaves[i].err for i in range(len(tree.leaves))])
            cost_complexities.append(tot_err + self.alpha * tree_size)
        best_idx = cost_complexities.index(min(cost_complexities))
        return pruned_list[best_idx]