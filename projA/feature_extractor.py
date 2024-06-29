import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer



#this runs when import
with open('vectorizer1.pkl', 'rb') as vec:
    vectorizer1 = pickle.load(vec)

def test_feature_extractor(text_list):
        # take the review sentence from each data point
        sentences = [sublist[1] for sublist in text_list]
        vec = CountVectorizer(vocabulary=vectorizer1.get_feature_names_out())
        X = vec.fit_transform(sentences)

        return X 