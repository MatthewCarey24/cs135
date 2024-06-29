# HW1: Regression, Cross-Validation, and Regularization


**Due date**: Tue. Feb. 8, 2024 by 11:59pm ET


## Overview

In this HW, you'll complete two parts in order:

* First, complete several *code tasks*, writing Python code based on provided starter code. You'll submit your code as a ZIP to the autograder link below.
* Second, complete some *data analysis* tasks (which use your code) and write a *report*. You'll submit this PDF report separately.

As much as possible, we have tried to *decouple* these parts, so you may successfully complete the report even if some of your code doesn't work.

**Turn-in links**:

* PDF report turned in to: <https://www.gradescope.com/courses/716369/assignments/3968960/>
* ZIP file of source code turned in to: <https://www.gradescope.com/courses/716369/assignments/3968962/>
* Then complete your reflection here: <https://forms.gle/gRPRxjDLVt6PPTTY8>

**Files to Turn In:**

ZIP file of source code (autograded) should contain *only* these files, without any folder structure:

* cross_validation.py
* performance_metrics.py
* LeastSquaresLinearRegression.py
* hw1_report.ipynb

PDF report (manually graded):

* Prepare a short PDF report (no more than 3 pages; ideally 1 page per problem below).
* Use your favorite report writing tool (Word or Docs or LaTeX or ....)
* Should be **human-readable**. Do not include code. Do NOT just export a jupyter notebook to PDF.
* Should have each subproblem [marked via the in-browser Gradescope annotation tool](https://www.youtube.com/watch?v=KMPoby5g_nE&feature=youtu.be&t=43))


**Evaluation Rubric:**

* 80% will be the report
* 18% will be the autograder score of your code
*  2% reflection

See the PDF submission portal on Gradescope for the point values of each problem. Generally, tasks with more coding/effort will earn more potential points.


## Background

To complete this HW, you'll need some specific knowledge from the following sessions of class:

* Training Linear Regression (day03)
* Polynomial Feature Transformations (day04)
* Cross Validation (day04)
* Regularized Linear Regression (day05)

# <a name="code-tasks">Code Tasks </a>

### <a name="starter-code"> Starter Code </a>

See the hw1 folder of the public assignments repo for this class:

*Will be released soon*

This starter code includes a notebook to help you organize your analysis, plus several `.py` files for core functionality you need to implement yourself.


### <a name="code-task-1"> Code Task 1: Edit `performance_metrics.py` to implement `calc_root_mean_squared_error` </a>


**Task 1(a)** : Implement `calc_root_mean_squared_error`

See the starter code for example inputs and the expected output.


### <a name="code-task-2"> Code Task 2: Edit `LeastSquaresLinearRegression.py` to implement `fit` and `predict` </a>


This file defines a `LeastSquaresLinearRegressor` class with the two key methods of the usual sklearn regression API: `fit` and `predict`. You will edit this file to complete the `fit` and the `predict` methods, which will demonstrate your understanding of what goes on "inside" sklearn-like regressor objects.

**Task 2(a)** : The `fit` method should take in a labeled dataset $$\{x_n, y_n\}_{n=1}^N$$ and instantiate two instance attributes


* `w_F` : 1D numpy array, shape (n_features = F,)
	Represents the 'weights'
	Contains float64 entries of the weight coefficients
* `b` : scalar float
	Represents the 'bias' or 'intercept'.

Hint: Within a Python class, you can set an attribute like `self.b = 1.0`. 

Nothing should be returned. You're updating the internal state of the object.

These attributes should be set using the formulas discussed in class (day03) for solving the "least squares" optimization problem (finding $w$ and $b$ values that minimize squared error on the training set).


**Task 2(b)** : The `predict` method** should take in an array of feature vectors $$\{x_n\}_{n=1}^N$$ and produce (return) the predicted responses $$\{ \hat{y}(x_n) \}_{n=1}^N$$

Recall that for linear regression, we've defined the prediction function as:

$$
\hat{y}(x_n) = b + w^T x_n = b + \sum_{f=1}^F w_f x_{nf}
$$


### <a name="code-task-3"> Code Task 3: Edit `cross_validation.py` to randomly divide data into splits and estimate training and heldout error </a>

**Task 3(a)** : Implement the `make_train_and_test_row_ids_for_n_fold_cv` function

This function should consume the number of examples, the desired number of folds, and a pseudo-random number generator.
Then, it will produce, for each of the desired number of folds, arrays of *integers* indicating which rows of the dataset belong to the training set, and which belong to the test set.

See the starter code for detailed specification.

Note : For each fold, you do NOT need to produce exactly the same random splits as our code. For instance, while creating 3 fold splits for an array all_examples=[1, 2, 3, 4, 5], the examples in each in fold could be : 

* fold0_examples=[1, 2]
* fold1_examples=[3, 5]
* fold2_examples=[4]

**OR**

* fold0_examples=[3, 4]
* fold1_examples=[1, 5]
* fold2_examples=[2]

**Task 3(b)** : Implement the `train_models_and_calc_scores_for_n_fold_cv` function

This function will use the procedure from 3(a) to determine the different "folds", and then train a separate model at each fold and return that model's training error and heldout error.

See the starter code for detailed specification.

# <a name="report-tasks">Report Tasks</a>

### <a name="dataset"> Dataset: Miles-per-Gallon efficiency of Vehicles </a>

You have been given a data set containing gas mileage, horsepower, and other information for 395 makes and models of vehicles.  For each vehicle, we have the following information:  

| column name        | type    | unit | description |
| -------------------- | ------- | ---- | ----------- |
| horsepower       | numeric | hp   | engine horsepower
| weight           | numeric | lb.  | vehicle weight
| cylinders        | numeric | #    | number of engine cylinders, from 4 to 8
| displacement     | numeric | cu. inches | overall volume of air inside engine
| mpg              | numeric | mi. / gal | vehicle miles per gallon

You have been asked to build a predictor for vehicle mileage (mpg) as a function of other vehicle characteristics. 

In the starter code, we have provided an existing train/validation/test split of this dataset, stored on-disk in comma-separated-value (CSV) files: x_train.csv, y_train.csv, x_valid.csv, y_valid.csv, x_test.csv, and y_test.csv.  



### <a name="problem-1">Problem 1: Polynomial Regression - Selecting Degree on a Fixed Validation Set </a>

For this problem, use the provided training set and validation set

* `x_train.csv` and `y_train.csv` contain features and outcomes for 192 examples
* `x_valid.csv` and `y_valid.csv` contain features and outcomes for 100 examples

Your goal is to determine which polynomial transformation yields the best predictive performance. We'll use **Root Mean Squared Error (RMSE)** throughout.

Follow the starter notebook. Your code should chain together the `PolynomialFeatures` and `LinearRegression` class provided by `sklearn`. 

**Implementation Step 1A:**
Fit a linear regression model to a polynomial feature transformation of the provided training set at each of these possible degrees: [1, 2, 3, 4, 5, 6, 7]. For each hyperparameter setting, record the training set error and the validation set error, in terms of RMSE.

**Implementation Step 1B:** 
Select the model hyperparameters that *minimize* your fixed validation set error. Using your already-trained LinearRegression model with these best hyperparameters, compute error on the *test* set. Save this test set error value for later.


**Figure 1 in Report:**

Make a line plot of RMSE on y-axis vs. polynomial degree on x-axis.
Show two lines, one for error on training set (in blue) and one for error on validation (in red). Follow style in provided starter notebook.

**Caption of Figure 1 in Report:**

Provide a 2 sentence caption answering the questions: Does this plot look as you expect based on course concepts? What degree do you recommend based on this plot? 


**Short Answer 1a in Report:**

The starter code pipelines include a *preprocessing* step that rescales each feature column to be in the unit interval from 0 to 1. 
Why is this necessary for this particular dataset?
What happens (in terms of both training error and test error) if this step is omitted?
*Hint: Try removing this step and see.*

**Short Answer 1b in Report:**

Consider the model with degree 1. 
Following the starter code, print out the values of **all** the learned weight parameters (aka coefficients).
From these values, what can you say about how increasing the engine weight impacts the prediction of mpg?
Does this make sense?
What about if you increase engine displacement? Recall that displacement refers to the overall volume of air that can move through the engine. Larger engine means larger displacement.

**Short Answer 1c in Report:**

Consider the models with degree 4 or larger.
Inspect the learned weight parameters, including the number of parameters and their relative magnitudes.
What do you notice about these values compared to the values with degree 1 or 2?
How might what you notice be connected to the trends in training and validation set performance you observe in Figure 1?




### <a name="problem-2">Problem 2: Penalized Polynomial Regression - Selecting Alpha on a Fixed Validation Set </a>

You should have noticed in problem 1 that models with large degree (like 4 or higher) show bad signs of overfitting. Your goal is to see if we can use a "ridge" penalty (introduced in day05) to smartly improve the heldout error of complex models (with degree 4).

Follow the starter notebook. Throughout problem 2, you should use the provided pipeline code to chain together the `PolynomialFeatures` and `Ridge` implementations provided by `sklearn`.

**Implementation Step 2A:**

Fix the degree at 4. Consider the following possible `alpha` values for penalized linear regression, aka `Ridge`:

```
alpha_list = np.asarray([1.e-10, 1.e-08, 1.e-06, 1.e-04, 1.e-02, 1.e+00, 1.e+02, 1.e+04, 1.e+06])
```

Fit a L2-penalized linear regression pipeline for each alpha value above, then record that model's training set error and the validation set error.


**Implementation Step 2B:** 

Select the model hyperparameters that *minimize* your fixed validation set error. Using your already-trained model with these best hyperparameters, compute error on the *test* set. Save this test set error value for later.


**Figure 2 in Report:**

Make a line plot of RMSE on y-axis vs. alpha on x-axis. Show two lines, one for error on training set (in blue) and one for error on validation (in red).

Follow the styles in starter code. Be sure to show alpha values on x-axis on log-scale.

**Caption of Figure 2 in Report:**

Provide a 2 sentence caption answering the questions: Does this plot look as you expect based on course concepts? What alpha value do you recommend based on this plot? 


**Short Answer 2a in Report**

Inspect the learned weight parameters of your chosen degree-4 model.
What do you notice about the relative magnitudes compared to 1c above?


**Short Answer 2b in Report:**

Your colleague suggests that you can determine the regularization strength `alpha` by minimizing the following loss on the *training* set:

$$
\text{min}_{w \in \mathbb{R}^F, b \in \mathbb{R}, \alpha \ge 0}
\quad \sum_{n=1}^N (y_n - \hat{y}(x_n, w, b))^2 + \alpha \sum_{f=1}^F w_f^2
$$

What value of $$\alpha$$ would you pick if you did this? Why is this problematic if your goal is to generalize to new data well?




### <a name="problem-3">Problem 3: Penalized Polynomial Regression + Model Selection with Cross-Validation </a>

For this problem, you'll again use the provided training set and validation sets. However, you'll *merge* these into a large "development" set that contains 292 examples total.


**Implementation Step 3A:**
Consider the following set of possible hyperparameters for a penalized ridge pipeline

```
degree_list = [1, 2, 3, 4, 5, 6, 7]
alpha_list = np.logspace(-10, 6, 17)
```

For each possible `alpha` value as well as each possible polynomial degree, train and evaluate a `Ridge` regression model across the entire train+validation set using 10-fold cross validation. Use the CV methods you implemented in `cross_validation.py`. For each possible hyperparameter configuration (alpha value and degree value), your 10-fold CV procedure will give you an estimate of the training error and heldout validation error across all K folds. Compute the mean validation error across all K folds to get your estimated cross-validation error. 

**Implementation Step 3B:**

Select the model hyperparameters that *minimize* your estimated cross-validation error. Using these best hyperparameters, retrain the model using the full development set (concatenating the predefined training and validation sets). Then compute that (retrained) model's error on the test set.
Save this test set error value for later.


**Table 3 in Report:**

In one neat table, please compare the *test set* root-mean-squared-error (RMSE) performance for the following regressors:

* Baseline: A predictor that always guesses the *mean* $$y$$ value of the training set, regardless of the new test input
* The best Poly+Linear pipeline, picking degree to minimize val set error (from 1B)
* The best Poly+Ridge pipeline, fixing degree=4 and picking alpha to minimize val set error (from 2B)
* The best Poly+Ridge pipeline, picking degree and alpha to minimize 10-fold cross validation error (from 3B)

**Caption of Table 3 in Report:**

Provide a 2 sentence caption answering the question: 
What method in Table 3 performs best in terms of heldout error?
Do the rankings of methods match what you expect based on course concepts?


### Helpful hints and best practices for preparing a report

Across all the problems here, be sure that:

* All plots include readable axes labels and legends if needed when multiple lines are shown.
* All figures include *captions* providing complete sentence summaries of the figure.
* Generally, all tables should only report floating-point values up to 3 decimal places in precision.
* * That is, if your error is 17.123456789, just display "17.123". Make it easy on your reader's eyes.
