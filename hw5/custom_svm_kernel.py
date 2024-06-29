import numpy as np
import sklearn.svm

def rbf_kernel(X1, X2, gamma):
    """
    Implementation of the RBF kernel. Please check the calculation in the sklearn documentation 
    <https://scikit-learn.org/stable/modules/svm.html#kernel-functions> 

    For a pair of instances $x_1$ and $x_2$, $rbf(x_1, x_2) = \exp(- \gamma * (x_1 - x_2)^\top (x_1 - x_2))$. 
    The implemenation should work for two instance matrices, that is, if K = rbf(X1, X2), 
    then K[i, j] should be the kernel value between X1[i] and X2[j]. It is strongly suggested to use matrix 
    computation, otherwise, the SVM with your kernel will be slow. 

    Args:
    ----
        X1: np.array with shape (N1, F), representing the first set of N1 instances. F is the number of features in X1
        X2: np.array with shape (N2, F), representing the second set of N2 instances
        gamma: a scalar, the hyperparameter of the kernel 

    Returns:
    ----
        K: np.array with shape (N1, N2), the RBF kernel between X1 and X2. 

    An example: run through an example with the iris dataset.  
    --------
    # load the data 
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X_NF, real_label = load_iris(return_X_y=True)
    >>> y_N = (real_label > 1).astype(np.int32)

    # train a standard SVM from sklearn. 
    # NOTE: the decision function computes the score z_n for an instance x_n. It is used
    # to decide the label y_n = 1 if z_n > 0 else 0
    >>> gamma = 1.0
    >>> svm_sk = SVC(kernel="rbf", gamma=gamma)
    >>> svm_sk = svm_sk.fit(X_NF, y_N) 
    >>> z_N_sk = svm_sk.decision_function(X_NF) 

    # We need to use a lambda function here to pass in the gamma parameter
    >>> svm_yours = SVC(kernel=lambda a, b: rbf_kernel(a, b, gamma=gamma))
    >>> svm_yours = svm_yours.fit(X_NF, y_N) 
    >>> z_N_yours = svm_yours.decision_function(X_NF) 

    # check the difference, which should be small
    # NOTE: the optimization problem of SVM is convex, which means that model training should 
    # always find the global optimal solution. Therefore, the two SVMs trained with the same settings
    # should be about the same. If not for the convex problem, these two may have large differences. 
    # For example, we cannot run such a test on MLPs. 

    >>> diff = np.mean(np.abs(z_N_sk - z_N_yours))
    >>> diff < 1e-6
    True
    >>> 


    """

    # TODO: implement your own version of RBF kernel. The calculation should follow
    # the documentation below.
    # <https://scikit-learn.org/stable/modules/svm.html#kernel-functions> 

    K = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i][j] = np.exp(-gamma * np.dot((X1[i]-X2[j]).T, (X1[i]-X2[j])))

    return K


