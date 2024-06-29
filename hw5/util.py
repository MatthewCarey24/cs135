from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay


def plot_training_data_with_decision_boundary(ax, title, clf, X, y, is_svm=False, is_regression=False):
    """
    Plot training data and the decision boundary of the classifier. If the model is SVM, 
    then also plot margins and support vectors. 

    args:
        ax: the handle of the subfigure
        title: string, the title of the plot 
        clf: an sklearn classifier
        X: np.array() with shape [N, F], the feature matrix of the training data 
        y: np.array() with shape [N, ], the label of the training data 
        is_svm: boolean, whether the model `clf` is an SVM

    returns:

    """

    x_min, x_max, y_min, y_max = 1.9, 4.5, -0.2, 2.7
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}

    levels = [-1, 0, 1] if is_svm else [0]
    linestyle = ["--", "-", "--"] if is_svm else ["-"]
    response = "decision_function" if not is_regression else "predict"

    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method=response,
        plot_method="contour",
        levels=levels,
        colors=["k", "k", "k"],
        linestyles=linestyle,
    )


    # Plot bigger circles around samples that serve as support vectors
    ax.scatter(X[:, 0], X[:, 1], c=y, s=8)
    if is_svm:
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=50, facecolors="none", edgecolors="k")
        
    # Plot samples by color and add legend
    ax.set_title(title)

