import numpy as np
import sklearn


# NOTE 1: in this problem, you are supposed to replicate the function:    
# `sklearn.tree.DecisionTreeClassifier.predict_proba`. Please check the 
# documentation of this function before your implementation. 

# NOTE 2: you CANNOT call `sklearn.tree.DecisionTreeClassifier.predict_proba`
# in your implementation. The purpose is to match your own implementation to 
# this function

# NOTE 3: `sklearn.tree.DecisionTreeClassifier.apply` is your friend and plays 
# a core role in your implementation. Please study its documentation. 



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
    leaf_pred : a dictionary like {leaf_id: probability_vector}
        each key is a leaf index, and the corresponding vector is the 
        probability vector for an instance in that leaf 
        
    '''


    leaf_pred = dict()

    # TODO Step 1: compute leaf indices for all instances.
    # You only need to call `sklearn.tree.DecisionTreeClassifier.apply` 
    # Please study the documentation before using this function 
    leaf_indices = tree_clf.apply(X_NF)


    # TODO Step 2: aggregate labels of instances in the same leaf 
    # For each leaf, you will need a vector with length C. Each 
    # entry of the vector contains the count of instances that 
    # fall in the leaf AND have the label corrresponding to the entry index  
    for leaf_index in np.unique(leaf_indices):
        mask = leaf_indices == leaf_index
        leaf_pred[leaf_index] = np.bincount(y_N[mask], minlength=len(np.unique(y_N)))

    
    # TODO Step 3: normalize the vector from step 2 to a probability vector  
        for leaf_index in leaf_pred:
            leaf_pred[leaf_index] = leaf_pred[leaf_index] / np.sum(leaf_pred[leaf_index])

    #C = y_N.max() + 1 # assume continuous label indices starting from 0

    return leaf_pred


def predict_proba(X_NF, tree_clf, leaf_pred):
    ''' Predicting labels for a set of instances using the tree structure `tree_clf` 
    and leaf predictions

    Args
    ----
    X_NF : 2D array, shape (N,F) = (n_examples, n_features)
        Training data features at current node we wish to find a split for.
    tree_clf: a tree classifier `sklearn.tree.DecisionTreeClassifier`
    leaf_pred: a dictionary like {leaf_id: probability_vector}
    Returns
    -------
    phat_NC : 2D array, shape (N, C)
        C is the number of classes. phat_NC should contain class probabilities, 
        each row for an instance.

    An example: run through an example with the iris dataset.  
    --------
    # load the data 
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> X_NF, y_N = load_iris(return_X_y=True)
    >>> C = y_N.max() + 1

    # use a shallow tree so each leaf node has non-extreme probabilities
    # get predictions from the sklearn
    >>> tree_clf = DecisionTreeClassifier(min_samples_split=30)
    >>> tree_clf = tree_clf.fit(X_NF, y_N)
    >>> sklearn_pred = tree_clf.predict_proba(X_NF)

    # get predictions from your implementation 
    >>> leaf_pred = fit_leaf_pred(X_NF, y_N, tree_clf)
    >>> your_pred = predict_proba(X_NF, tree_clf, leaf_pred)

    # check the difference between your predictions and sklearn's predictions
    # the difference should be less than 1e-6
    >>> diff = np.mean(np.abs(sklearn_pred - your_pred))
    >>> diff < 1e-6
    True
    >>> 

    # check whether your code actually uses leaf_pred 
    >>> bad_leaf_pred = dict((leaf, leaf_pred[leaf] + 1.0) for leaf in leaf_pred.keys())
    >>> bad_pred = predict_proba(X_NF, tree_clf, bad_leaf_pred)
    >>> diff = np.mean(np.abs(bad_pred - your_pred))
    >>> diff > 1e-2 
    True
    >>>
    '''

    leaf_indices = tree_clf.apply(X_NF)
    pred = np.array([leaf_pred[leaf_index] for leaf_index in leaf_indices])

    # TODO: first get leaf indices of instances in X, and then look up corresponding 
    # predictions from leaf_pred

    return pred
