import numpy as np
import sklearn


# NOTE 1: in this problem, you are supposed to replicate the function:    
# `sklearn.tree.DecisionTreeRegressor.predict`. Please check the 
# documentation of this function before your implementation. 

# NOTE 2: you CANNOT call `sklearn.tree.DecisionTreeRegressor.predict`
# in your implementation. The purpose is to match your own implementation to 
# this function

# NOTE 3: `sklearn.tree.DecisionTreeClassifier.apply` plays a core role 
# in your implementation. Please study its documentation. 


def fit_leaf_pred(X_NF, y_N, tree_clf):

    '''
    Fitting leaf predictions from the training set. Here we assume that we only 
    have the trained structure. 

    Args
    ----
    X_NF : 2D array, shape (N,F) = (n_examples, n_features)
        Training data features at current node we wish to find a split for.
    y_N : 1D array, shape (N,) = (n_examples,)
        Training labels at current node.
    Returns
    -------
    leaf_pred : a dictionary like {leaf_id: pred}
        each key is a leaf index, and the corresponding prediction is the 
        scalar prediction for an instance in that leaf 
        
    '''

    leaf_pred = dict()

    # TODO Step 1: compute leaf indices for all instances.
    # You only need to call `sklearn.tree.DecisionTreeRegressor.apply` 
    # Please study the documentation before using this function 
    leaf_indices = tree_clf.apply(X_NF)


    # TODO Step 2: aggregate labels of instances in the same leaf 
    # By default the model minimizes the square error, so the prediction
    # should be the mean of the labels in the leaf node. 
    for leaf_index in np.unique(leaf_indices):
        mask = leaf_indices == leaf_index
        leaf_pred[leaf_index] = np.mean(y_N[mask])

    
    return leaf_pred


def predict(X_NF, tree_clf, leaf_pred):
    ''' Predicting labels for a set of instances using the tree structure `tree_clf` 
    and leaf predictions

    Args
    ----
    X_NF : 2D array, shape (N,F) = (n_examples, n_features)
        Training data features at current node we wish to find a split for.
    tree_clf: a tree classifier `sklearn.tree.DecisionTreeRegressor`
    leaf_pred: a dictionary like {leaf_id: pred}
    Returns
    -------
    yhat_N : 2D array, shape (N, )
        phat_NC should contain scalar predictions, each entry for an instance.

    An example: run through an example with the diabetes dataset.  
    --------
    # load the data 
    >>> from sklearn.datasets import load_diabetes 
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> X_NF, y_N = load_diabetes(return_X_y=True)

    # use a shallow tree so each leaf node has non-extreme probabilities
    # get predictions from the sklearn
    >>> tree_clf = DecisionTreeRegressor(min_samples_split=30)
    >>> tree_clf = tree_clf.fit(X_NF, y_N)
    >>> sklearn_pred = tree_clf.predict(X_NF)

    # get predictions from your implementation 
    >>> leaf_pred = fit_leaf_pred(X_NF, y_N, tree_clf)
    >>> your_pred = predict(X_NF, tree_clf, leaf_pred)

    # check the difference between your predictions and sklearn's predictions
    # the difference should be less than 1e-6
    >>> diff = np.mean(np.abs(sklearn_pred - your_pred))
    >>> diff < 1e-6
    True
    >>> 

    # check whether your code actually uses leaf_pred 
    >>> bad_leaf_pred = dict((leaf, leaf_pred[leaf] + 10.0) for leaf in leaf_pred.keys())
    >>> bad_pred = predict(X_NF, tree_clf, bad_leaf_pred)
    >>> diff = np.mean(np.abs(bad_pred - your_pred))
    >>> diff > 1e-2 
    True
    >>>
    '''

    pred = None

    # TODO: first get leaf indices of instances in X, and then look up corresponding 
    # predictions from leaf_pred
    leaf_indices = tree_clf.apply(X_NF)
    pred = np.array([leaf_pred[leaf_index] for leaf_index in leaf_indices])

    return pred
