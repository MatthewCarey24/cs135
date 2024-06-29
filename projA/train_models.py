import numpy as np
import pandas as pd
import os
import pickle

import sklearn.pipeline
import sklearn.linear_model
import sklearn.metrics
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


'''Use This class for building our classifiers
    --> bring in our bag of words and feature extractor for model building
'''
def overview_data(x_train_df, y_train_df):

    N, n_cols = x_train_df.shape
    print("Shape of x_train_df: (%d, %d)" % (N,n_cols))
    print("Shape of y_train_df: %s" % str(y_train_df.shape))

    # print a row in the dataframe as a list of two strings: website and the review. 
    print("\nPrint three instance from the training set in lists:")
    print("Input (website, review):")
    print((x_train_df.iloc[0:3, :]).values.tolist())
    print("Label:")
    print(y_train_df.iloc[0:3, 0].tolist())

    # Print out the first five rows and last five rows
    print("\n")
    print("More data from training set:")
    tr_text_list = x_train_df['text'].values.tolist()
    rows = np.arange(0, 5)
    for row_id in rows:
        text = tr_text_list[row_id]
        print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))

    print("...")
    rows = np.arange(N - 5, N)
    for row_id in rows:
        text = tr_text_list[row_id]
        print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))

def feature_extractor1(text_list):
        # take the review from each document
        sentences = [sublist[1] for sublist in text_list]
        # use a count vectorizer 
        vec = CountVectorizer(max_df=0.8, lowercase=True, analyzer='word', ngram_range=(1,1))
        # transform the sentences into a bag of words rep
        BOW = vec.fit_transform(sentences)
        #return our BOW and vectorizer
        return BOW, vec

def main(data_dir='data_reviews'):
    # overview the training data
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    overview_data(x_train_df, y_train_df)
    x_train_text = x_train_df.values.tolist()
    BOW, vectorizer = feature_extractor1(x_train_text)
    
    assert x_train_df.shape[0] == y_train_df.shape[0]

    # Parameters to check
    param_grid1 = {
        'C': np.logspace(-9, 6, 31),
        'penalty': ['l2']
    }

    param_grid2 = {
    'n_estimators': [100, 300],
    'max_depth': [None, 8],
    'max_features': ['log2', 8]
    }

    # plot_heldout_performance(BOW, y_train_df.values.ravel(), param_grid1)
    plot_perf_2(BOW, y_train_df.values.ravel(), param_grid2)

    # logreg = RandomForestClassifier()
    # #Define k-fold using SKlearn
    # k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    # '''--------------------------------------------------
    # Use conveinent SKLearn grid search for hyperparameters
    # # Estimator: logistic regression (sklearn)
    # # Parameters: above
    # # CV Scheme: k_fold (sklearn)
    # # Score Function : AUROC by piazza description'''
    # grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid2, cv=k_fold, scoring='roc_auc')
    # grid_search.fit(BOW, y_train_df.values.ravel())

    # # print("Best hyperparameters:", grid_search.best_params_)
    # # obtain best model from grid search
    # best_model = grid_search.best_estimator_
    # y_proba = best_model.predict_proba(BOW)[:,1]
    # #output roc score of our model
    # # print(roc_auc_score(y_train_df, y_proba))

    # # Assuming you have already trained your logistic regression model and stored it in 'best_model'
    # # Make predictions on the training data
    # y_train_pred = best_model.predict(BOW)

    # # Generate confusion matrix
    # conf_matrix = confusion_matrix(y_train_df.values.ravel(), y_train_pred)

    # # Print the confusion matrix
    # print("Confusion Matrix:")
    # print(conf_matrix)

    # # Assuming you have access to your original training data X_train and y_train_pred

    # # Get indices of true positive, false positive, true negative, and false negative predictions
    # tp_indices = np.where((y_train_df.values.ravel() == 1) & (y_train_pred == 1))[0]
    # fp_indices = np.where((y_train_df.values.ravel() == 0) & (y_train_pred == 1))[0]
    # tn_indices = np.where((y_train_df.values.ravel() == 0) & (y_train_pred == 0))[0]
    # fn_indices = np.where((y_train_df.values.ravel() == 1) & (y_train_pred == 0))[0]

    # # Print examples for each category
    # print("True Positives:")
    # for idx in tp_indices[:10]:  # Print the first 5 examples
    #     print(x_train_text[idx], y_train_pred[idx])

    # print("\nFalse Positives:")
    # for idx in fp_indices[:10]:
    #     print(x_train_text[idx], y_train_pred[idx])

    # print("\nTrue Negatives:")
    # for idx in tn_indices[:10]:
    #     print(x_train_text[idx], y_train_pred[idx])

    # print("\nFalse Negatives:")
    # for idx in fn_indices[:10]:
    #     print(x_train_text[idx], y_train_pred[idx])


    
    # # dump our best model we found to classifier 1
    # with open('classifier1.pkl', 'wb') as f:
    #     pickle.dump(best_model, f)    
    # # Pickling the vectorizer to a file
    # with open('vectorizer1.pkl', 'wb') as f:
    #     pickle.dump(vectorizer, f)








def plot_heldout_performance(X_train, y_train, param_grid):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Lists to store results
    hyperparam_values = []
    avg_heldout_losses = []
    heldout_losses_folds = []
    training_losses = []

    for C in param_grid['C']:
        for penalty in param_grid['penalty']:
            # Skip if using L-BFGS solver with L1 penalty
            if penalty == 'l1' and solver == 'lbfgs':
                continue
            
            # Set hyperparameters
            model = LogisticRegression(C=C, penalty=penalty, solver='lbfgs')
            
            # Perform cross-validation
            heldout_losses = []
            for train_index, val_index in k_fold.split(X_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                
                model.fit(X_train_fold, y_train_fold)
                y_val_proba = model.predict_proba(X_val_fold)[:, 1]
                heldout_loss = roc_auc_score(y_val_fold, y_val_proba)
                heldout_losses.append(heldout_loss)
                
            
            # Store results
            hyperparam_values.append((C, penalty))
            avg_heldout_losses.append(np.mean(heldout_losses))
            heldout_losses_folds.append(heldout_losses)
    bestIndex = avg_heldout_losses.index(np.max(avg_heldout_losses))
    print("Best C: ", hyperparam_values[bestIndex])
    print("Best AUROC: ", avg_heldout_losses[bestIndex])
    # Plot the results
    plt.figure(figsize=(10, 6))
    for i, (C, penalty) in enumerate(hyperparam_values):
        plt.plot([np.log10(C)] * len(heldout_losses_folds[i]), heldout_losses_folds[i], 'bo', alpha=0.5, label='Validation AUROC for Each Fold' if i == 0 else '')
        plt.plot(np.log10(C), avg_heldout_losses[i], 'ro', label='Average Validation AUROC' if i == 0 else '')
    plt.xticks(np.log10(param_grid['C']), [str(int(np.log10(C))) for C in param_grid['C']])
    plt.xlabel('log(C)')
    plt.ylabel('AUROC')
    plt.title('AUROC for Different Hyperparameter Values')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# def plot_perf_2(X_train, y_train, param_grid):
#     k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

#     # Lists to store results
#     hyperparam_values = []
#     avg_heldout_losses = []
#     heldout_losses_folds = []

#     for n_estimators in param_grid['n_estimators']:
#         # Set hyperparameters


#         model = RandomForestClassifier(n_estimators=n_estimators)
        
#         # Perform cross-validation
#         heldout_losses = []
#         for train_index, val_index in k_fold.split(X_train):
#             X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
#             y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
#             model.fit(X_train_fold, y_train_fold)
#             y_val_proba = model.predict_proba(X_val_fold)[:, 1]
#             heldout_loss = roc_auc_score(y_val_fold, y_val_proba)
#             heldout_losses.append(heldout_loss)
            
        
#         # Store results
#         hyperparam_values.append(n_estimators)
#         avg_heldout_losses.append(np.mean(heldout_losses))
#         heldout_losses_folds.append(heldout_losses)
    
#     bestIndex = avg_heldout_losses.index(np.max(avg_heldout_losses))
#     print("Best n_estimators: ", hyperparam_values[bestIndex])
#     print("Best AUROC: ", avg_heldout_losses[bestIndex])
    
#     # Plot the results
#     plt.figure(figsize=(10, 6))
#     for i, n_estimators in enumerate(hyperparam_values):
#         plt.plot([n_estimators] * len(heldout_losses_folds[i]), heldout_losses_folds[i], 'bo', alpha=0.5, label='Validation AUROC for Each Fold' if i == 0 else '')
#         plt.plot(n_estimators, avg_heldout_losses[i], 'ro', label='Average Validation AUROC' if i == 0 else '')
#     plt.xlabel('n_estimators')
#     plt.ylabel('AUROC')
#     plt.title('AUROC for Different n_estimators Values')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
def plot_perf_2(X_train, y_train, param_grid):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Lists to store results
    hyperparam_values = []
    avg_heldout_losses = []
    heldout_losses_folds = []

    # Generate all combinations of hyperparameters
    param_combinations = itertools.product(param_grid['n_estimators'],
                                            param_grid['max_depth'],
                                            param_grid['max_features'])

    for n_estimators, max_depth, max_features in param_combinations:
        # Set hyperparameters
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       max_features=max_features)
        
        # Perform cross-validation
        heldout_losses = []
        for train_index, val_index in k_fold.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            model.fit(X_train_fold, y_train_fold)
            y_val_proba = model.predict_proba(X_val_fold)[:, 1]
            heldout_loss = roc_auc_score(y_val_fold, y_val_proba)
            heldout_losses.append(heldout_loss)
            
        
        # Store results
        hyperparam_values.append((n_estimators, max_depth, max_features))
        avg_heldout_losses.append(np.mean(heldout_losses))
        heldout_losses_folds.append(heldout_losses)
    
    # Find the best hyperparameters
    bestIndex = avg_heldout_losses.index(np.max(avg_heldout_losses))
    best_hyperparams = hyperparam_values[bestIndex]
    print("Best hyperparameters:", best_hyperparams)
    print("Best AUROC:", avg_heldout_losses[bestIndex])
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    for i, hyperparams in enumerate(hyperparam_values):
        plt.plot([i] * len(heldout_losses_folds[i]), heldout_losses_folds[i], 'bo', alpha=0.5, label='Validation AUROC for Each Fold' if i == 0 else '')
        plt.plot(i, avg_heldout_losses[i], 'ro', label='Average Validation AUROC' if i == 0 else '')
    plt.xticks(range(len(hyperparam_values)), [str(hyperparams) for hyperparams in hyperparam_values], rotation=90)
    plt.xlabel('Hyperparameters')
    plt.ylabel('AUROC')
    plt.title('AUROC for Different Hyperparameter Combinations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()






    
if __name__ == '__main__':
    main()
