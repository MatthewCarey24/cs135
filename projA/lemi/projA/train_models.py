import numpy as np
import pandas as pd
import os
import pickle

import sklearn.linear_model
import sklearn.metrics
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier





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
    
def feature_extractor2(text_list):
        # take the review from each document
        sentences = [sublist[1] for sublist in text_list]
        # use a TFIDF vectorizer 
        vec = TfidfVectorizer(max_df=0.8, lowercase=True, analyzer='word', ngram_range=(1,2), smooth_idf=True)
        # transform the sentences into a bag of words rep
        BOW = vec.fit_transform(sentences)
        #return our BOW and vectorizer
        return BOW, vec

def main(data_dir='data_reviews'):
    # overview the training data
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    overview_data(x_train_df, y_train_df)
    train_model1(x_train_df, y_train_df)
    train_model2(x_train_df, y_train_df)
    

def train_model1(x_train_df, y_train_df):
    x_train_text = x_train_df.values.tolist()
    BOW, vectorizer = feature_extractor1(x_train_text)
    
    assert x_train_df.shape[0] == y_train_df.shape[0]

    # Parameters to check
    param_grid = {
        'C': np.logspace(-9, 6, 31),
        'penalty': ['l2']
    }
    logreg = LogisticRegression(solver='lbfgs', max_iter=10000)
    #Define k-fold using SKlearn
    k_fold = KFold(n_splits=5, shuffle=True, random_state=20)
    '''--------------------------------------------------
    Use conveinent SKLearn grid search for hyperparameters
    # Estimator: logistic regression (sklearn)
    # Parameters: above
    # CV Scheme: k_fold (sklearn)
    # Score Function : AUROC by piazza description'''
    grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=k_fold, scoring='roc_auc')    
    grid_search.fit(BOW, y_train_df.iloc[:, 0].values)
    
    print("Best hyperparameters:", grid_search.best_params_)
    
    # Obtain the best model from the grid search
    best_model = grid_search.best_estimator_
    
    # print("Best hyperparameters:", grid_search.best_params_)
    y_proba = best_model.predict_proba(BOW)[:,1]
    #output roc score of our model
    # print(roc_auc_score(y_train_df, y_proba))
    
    # dump our best model we found to classifier 1
    with open('classifier1.pkl', 'wb') as f:
        pickle.dump(best_model, f)    
    # Pickling the vectorizer to a file
    with open('vectorizer1.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)



# def train_model2(x_train_df, y_train_df):
#     assert x_train_df.shape[0] == y_train_df.shape[0]
    
#     x_train_text = x_train_df.values.tolist()
#     BOW, vectorizer = feature_extractor2(x_train_text)
    
#     # Parameters to check
#     param_grid = {
#         'C': np.logspace(-9, 6, 31),
#         'penalty': ['l2']
#     }
#     logreg = LogisticRegression(solver='lbfgs', max_iter=10000)
#     #Define k-fold using SKlearn
#     k_fold = KFold(n_splits=5, shuffle=True, random_state=20)
#     '''--------------------------------------------------
#     Use conveinent SKLearn grid search for hyperparameters
#     # Estimator: logistic regression (sklearn)
#     # Parameters: above
#     # CV Scheme: k_fold (sklearn)
#     # Score Function : AUROC by piazza description'''
#     grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=k_fold, scoring='roc_auc')    
#     grid_search.fit(BOW, y_train_df.iloc[:, 0].values)
    
#     print("Best hyperparameters:", grid_search.best_params_)
    
#     # Obtain the best model from the grid search
#     best_model = grid_search.best_estimator_
    
#     # Optionally: Save the best model and the entire pipeline (including vectorizer)
#     with open('classifier2.pkl', 'wb') as f:
#         pickle.dump(best_model, f)
        
#     with open('vectorizer2.pkl', 'wb') as f:
#         pickle.dump(vectorizer, f)
    

def train_model2(x_train_df, y_train_df):
    assert x_train_df.shape[0] == y_train_df.shape[0]
    
    x_train_text = x_train_df.values.tolist()
    BOW, vectorizer = feature_extractor2(x_train_text)
    
    # Define classifiers and their respective parameter grids
    classifiers = {
        'Random Forest': (RandomForestClassifier(max_features=8, max_depth = 8, n_jobs=-1), 
                          {'n_estimators': [1000]}),
        'MLP Classifier': (MLPClassifier(hidden_layer_sizes=10),
                            {'alpha': np.logspace(-6, 6, 20)}),
    
  
            'Logistic Regression': (LogisticRegression(solver='lbfgs', max_iter=10000), 
                                {'C': np.logspace(-9, 8, 50), 'penalty': ['l2']}),
    }
    best_model = None
    best_score = 0
    
    # Iterate over classifiers
    for clf_name, (clf, param_grid) in classifiers.items():
        #Define k-fold using SKlearn
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
        print("\n\n\n\n here \n\n\n")
        # Grid search for hyperparameters
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=k_fold, scoring='roc_auc')
        grid_search.fit(BOW, y_train_df.iloc[:, 0].values)
        
        # Print best hyperparameters
        print(f"Best hyperparameters for {clf_name}: {grid_search.best_params_}")
        
        # Check if this model has better performance
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
        
    print("best model:" + grid_search.best_estimator_.__class__.__name__)
    with open('classifier2.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    with open('vectorizer2.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

 
if __name__ == '__main__':
    main()
