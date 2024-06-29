'''
proba_metrics.py

Provides implementation of common metrics for assessing a binary classifier's
*probabilistic* predictions against true binary labels, including:

* binary cross entropy from probabilities
* binary cross entropy from scores (real-values, to be fed into sigmoid)
'''

import numpy as np

from scipy.special import logsumexp as scipy_logsumexp
from scipy.special import expit as sigmoid

def calc_mean_binary_cross_entropy_from_probas(ytrue_N, yproba1_N):
    ''' Compute average cross entropy for given binary classifier's predictions

    Consumes probabilities ("probas"), values between 0.0 and 1.0.

    Computing BCE uses *base-2* logarithms, so the resulting number is a valid
    upper bound of the zero-one loss (aka error rate) when we threshold at 0.5.

    Notes
    -----
    Given a binary label $y_n \in \{0, 1}$ and a probability $p_n \in (0,1)$,
    we define binary cross entropy as:
    $$
        BCE(y_n, p_n) = - y_n \log_2 p_n - (1-y_n) \log_2 (1-p_n)
    $$
    Given $N$ labels and their predicted probas, we define the mean BCE as:
    $$
        mean_BCE(y, p) = \frac{1}{N} \sum_{n=1}^N BCE(y_n, p_n)
    $$

    Args
    ----
    ytrue_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yproba1_N : 1D array, shape (n_examples,) = (N,)
        All values must be within the interval 0.0 - 1.0, inclusive.
        Will be truncated to (eps, 1 - eps) to keep log values from extremes,
        with small value eps equal to 10^{-14}.
        Each entry is probability that that example should have positive label.
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    bce : float
        Binary cross entropy, averaged over all N provided examples

    Examples
    --------
    >>> N = 8
    >>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])

    # Try perfect predictions
    >>> perfect_proba1_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
    >>> perfect_bce = calc_mean_binary_cross_entropy_from_probas(
    ...     ytrue_N, perfect_proba1_N)
    >>> print("%.4f" % perfect_bce)
    0.0000

    # Try some confident scores
    >>> good_proba1_N = np.asarray([0.01, 0.05, 0.1, 0.1, 0.9, 0.9, 0.95, 0.97])
    >>> good_bce = calc_mean_binary_cross_entropy_from_probas(
    ...     ytrue_N, good_proba1_N)
    >>> print("%.4f" % good_bce)
    0.1018

    # Try some decent but underconfident scores
    >>> ok_proba1_N = np.asarray([0.3, 0.4, 0.46, 0.47, 0.5, 0.6, 0.7, 0.71])
    >>> ok_bce = calc_mean_binary_cross_entropy_from_probas(
    ...     ytrue_N, ok_proba1_N)
    >>> print("%.4f" % ok_bce)
    0.7253

    # Try some mistakes that are way over confident
    >>> bad_pr1_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
    >>> bad = calc_mean_binary_cross_entropy_from_probas(ytrue_N, bad_pr1_N)
    >>> print("%.4f" % bad)
    17.4403

    # Try empty
    >>> empty_bce = calc_mean_binary_cross_entropy_from_probas([], [])
    >>> np.allclose(0.0, empty_bce)
    True
    '''

    # Cast labels to integer just to be sure we're getting what's expected
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    N = int(ytrue_N.size)

    if(N == 0):
        return 0.0

    # Cast probas to float and be sure we're between zero and one
    yproba1_N = np.asarray(yproba1_N, dtype=np.float64)           # dont touch
    yproba1_N = np.maximum(1e-14, np.minimum(1-1e-14, yproba1_N)) # dont touch



    # be sure to handle empty input lists properly
    bce_prob = (-ytrue_N * np.log2(yproba1_N) - (1 - ytrue_N) * np.log2(1 - yproba1_N)).mean()

    return bce_prob


def calc_mean_binary_cross_entropy_from_scores(ytrue_N, scores_N):
    ''' Compute average cross entropy for given binary classifier's predictions

    Consumes "scores", real values between (-np.inf, np.inf),
    and then conceptually produces probabilities by feeding into sigmoid,
    and then into the BCE function to get the result.

    In practice, we compute BCE directly from scores using an implementation
    that avoids the numerical issues of the sigmoid (saturation, if underflows
    to zero would become negative infinity). To avoid possible numerical issues,
    you need to use the "logsumexp" trick, which is implemented by either
    `numpy.logaddexp` or `scipy.special.logsumexp`.

    Computing BCE uses *base-2* logarithms, so the resulting number is a valid
    upper bound of the zero-one loss (aka error rate) when we threshold at 0.5.

    Args
    ----
    ytrue_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    scores_N : 1D array, shape (n_examples,) = (N,)
        One entry per example in current dataset.
        Each entry is a real value, could be between -infinity and +infinity.
        Large negative values indicate strong probability of label y=0.
        Zero values indicate probability of 0.5.
        Large positive values indicate strong probability of label y=1.
        Needs to be same size as ytrue_N.

    Returns
    -------
    bce : float
        Binary cross entropy, averaged over all N provided examples
    '''
    # Cast labels to integer just to be sure we're getting what's expected
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    N = ytrue_N.size

    if N == 0:
        return 0.0

    # Convert binary y values so 0 becomes +1 and 1 becomes -1
    # See HW2 instructions on website for the math
    yflippedsign_N = -1 * np.sign(ytrue_N - 0.001)  # dont touch

    # Cast logit scores to float
    scores_N = np.asarray(scores_N, dtype=np.float64)  # dont touch

    flipped_scores_N = yflippedsign_N * scores_N  # fix me: flip(y_n) s_n

    scores_and_zeros_N2 = np.column_stack((np.zeros(N), flipped_scores_N))  # fix me: [0, flipped_scores_N]

    # Be sure to use `numpy.logaddexp` or `scipy.special.logsumexp` to handle 2D arrays and handle empty input lists properly
    bce_score = np.mean(scipy_logsumexp(scores_and_zeros_N2 + 1e-10, axis = 1) / np.log(2))

    return bce_score




# ytrue_N=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,]
# scores_N=[-2, -2, -2, -2, -2, -2, -2, -1,  1,  3,  5,  7,  9,]
# print(calc_mean_binary_cross_entropy_from_scores(ytrue_N, scores_N))
































####################################################################################################
# This is the 2024S version of this assignment. Please do not remove or make changes to this block.# 
# Otherwise, you submission will be viewed as files copied from other resources.                   # 
####################################################################################################




