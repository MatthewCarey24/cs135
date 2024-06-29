# This file is suppose to provide an interface between your implementation and the autograder. 
# In reality, the autograder should be a production system. This file provide an interface for 
# the system to call your classifier. 

# Ideally you could bundle your feature extractor and classifier in a single python function, 
# which takes a raw instance (a list of two strings) and predict a probability. 

# Here we use a simpler interface and provide the feature extractor and the classifer separately. 
# For Problem 1, you are supposed to provide
# * a feature extraction function `extract_BoW_features`, and  
# * a sklearn classifier, `classifier1`, whose `predict_proba` will be called.
# * your team name

# These two python objects will be imported by the `test_classifier_before_submission` autograder.

import pickle
import pandas as pd

from feature_extractor import test_feature_extractor1

with open('classifier1.pkl', 'rb') as f:
    classifier1 = pickle.load(f)
# with open('stub\\vectorizer1.pkl', 'rb') as vec:
#     vectorizer1 = pickle.load(vec)

# vocab = vectorizer1.get_feature_names_out()
extract_BoW_features = test_feature_extractor1

teamname = "PrincesofPython"

