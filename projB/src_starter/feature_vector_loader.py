import pandas as pd
import os

def load_features():
    data_path = '..\\data_movie_lens_100k'
    user_features = pd.read_csv(os.path.join(data_path,'user_info.csv'))
    item_features = pd.read_csv(os.path.join(data_path, 'movie_info.csv'))
    return user_features, item_features

