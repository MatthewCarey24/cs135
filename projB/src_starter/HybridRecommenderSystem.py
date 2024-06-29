from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
from train_valid_test_loader import load_train_valid_test_datasets
from feature_vector_loader import load_features
import autograd.numpy as ag_np
import pandas as pd


class HybridRecommenderSystem(CollabFilterOneVectorPerItem):
    def __init__(self, user_features, item_features, **kwargs):
        super().__init__(**kwargs)
        self.user_features = user_features
        self.item_features = item_features




    def prepare_features(self, user_ids, item_ids):
        latent_vectors_predictions = super().predict(user_ids, item_ids)  # Using CollabFilterOneVectorperItem's predict
        latent_vectors_predictions = latent_vectors_predictions.reshape(-1, 1) 

        user_feats = self.user_features.loc[user_ids].values
        # Only include numerical values as features (try BoW later)
        item_feats = self.item_features.loc[item_ids].select_dtypes(include=[ag_np.number]).values
        # stop_words = ['the', 'in', 'it', 'to', 'of', 'a', 'i', 'ii', 'iii', 'about', 'all', 'and', 'for', 'with', 'first', 'away', 'bad', 'before', 'being']
        # stop_words = ['the', 'about', 'above', 'adventure', 'adventures', 'affair', 'again', 'age', 'air', 'all', 'alone', 'america', 'american', 'amy', 'and', 'andy', 'angel', 'angels', 'anna', 'apartment', 'april', 'army', 'august', 'away', 'baby', 'back', 'bad', 'bananas', 'beach', 'beast', 'beautiful', 'beauty', 'before', 'being', 'bell', 'belle', 'beloved', 'best', 'beverly', 'bewegte', 'beyond', 'big', 'bitter', 'black', 'blade', 'blood', 'blue', 'board', 'body', 'book', 'bound', 'boy', 'boys', 'brady', 'broken', 'bronx', 'brother', 'brown', 'bulletproof', 'bunch', 'burnt', 'butcher', 'butterfly', 'call', 'calls', 'candidate', 'cape', 'cats', 'cause', 'chairman', 'chase', 'chasing', 'child', 'children', 'chocolate', 'circle', 'citizen', 'city', 'clean', 'clear', 'clockwork', 'close', 'club', 'cold', 'colors', 'company', 'conqueror', 'contact', 'control', 'cool', 'cop', 'country', 'courage', 'crazy', 'creatures', 'crossing', 'crow', 'crypt', 'cure', 'curse', 'dance', 'dangerous', 'dark', 'darkness', 'darn', 'das', 'day', 'days', 'dead', 'deadly', 'dear', 'death', 'deceiver', 'der', 'des', 'desert', 'designated', 'desire', 'desperate', 'devil', 'die', 'dirty', 'dog', 'dogs', 'dollhouse', 'don', 'doors', 'double', 'down', 'dracula', 'dream', 'dreams', 'drop', 'dumbo', 'duty', 'earth', 'engel', 'english', 'escape', 'est', 'eve', 'even', 'evil', 'eye', 'face', 'faces', 'fair', 'fall', 'falling', 'falls', 'family', 'farewell', 'father', 'favorite', 'fear', 'feeling', 'fell', 'final', 'fire', 'first', 'fish', 'flesh', 'fly', 'fool', 'for', 'forbidden', 'foreign', 'four', 'free', 'french', 'friday', 'friends', 'from', 'frontier', 'full', 'funeral', 'game', 'gang', 'garden', 'generation', 'george', 'get', 'getting', 'ghost', 'giant', 'gift', 'girl', 'girls', 'glory', 'god', 'gold', 'golden', 'gone', 'good', 'grand', 'grave', 'great', 'green', 'grosse', 'guns', 'hall', 'happened', 'hard', 'harlem', 'hat', 'hate', 'head', 'heart', 'heaven', 'heavenly', 'heavy', 'her', 'hero', 'high', 'highlander', 'hill', 'hills', 'his', 'hollywood', 'home', 'homeward', 'hood', 'horizon', 'horse', 'hot', 'hour', 'house', 'how', 'hugo', 'human', 'hunt', 'hurricane', 'ice', 'iii', 'indian', 'innocent', 'instinct', 'island', 'jack', 'jackie', 'jane', 'jaws', 'jimmy', 'johnny', 'jones', 'journey', 'judgment', 'jungle', 'jurassic', 'jury', 'kid', 'kids', 'kill', 'killer', 'killers', 'killing', 'king', 'kiss', 'knight', 'know', 'kombat', 'kull', 'lady', 'land', 'last', 'law', 'lawnmower', 'legend', 'les', 'letter', 'lies', 'life', 'like', 'line', 'little', 'live', 'living', 'long', 'lord', 'losing', 'lost', 'love', 'lover', 'loves', 'low', 'mad', 'made', 'madison', 'madness', 'magic', 'man', 'manhattan', 'mann', 'mary', 'mask', 'measures', 'meet', 'men', 'menace', 'metal', 'michael', 'midnight', 'mighty', 'minds', 'money', 'monkey', 'moon', 'mother', 'mountain', 'mourner', 'mouth', 'movie', 'mrs', 'much', 'murder', 'murders', 'music', 'mystery', 'naked', 'natural', 'never', 'new', 'next', 'night', 'nights', 'nightwatch', 'nobody', 'noon', 'north', 'not', 'nothing', 'now', 'off', 'old', 'once', 'one', 'only', 'other', 'out', 'over', 'own', 'panther', 'paradise', 'paris', 'part', 'parts', 'party', 'peak', 'people', 'perfect', 'personal', 'philadelphia', 'picture', 'pink', 'pool', 'power', 'presents',  'prisoner', 'private', 'prophecy', 'queen', 'quiet', 'race', 'rain', 'rangers', 'red', 'remains', 'return', 'rich', 'richard', 'rising', 'river', 'road', 'robin', 'robinson', 'rock', 'rocket', 'romeo', 'roof', 'room', 'rooms', 'rose', 'sabrina', 'safe', 'saint', 'santa', 'scarlet', 'school', 'scream', 'sea', 'search', 'season', 'secret', 'sense', 'serenade', 'seven', 'sex', 'shadow', 'shadows', 'shall', 'she', 'short', 'show', 'siege', 'silence', 'simple', 'sky', 'slate', 'sleep', 'sliding', 'sling', 'smoke', 'snatchers', 'snow', 'society', 'some', 'son', 'space', 'speed', 'spirits', 'spy', 'squeeze', 'stand', 'stars', 'stone', 'storm', 'story', 'stranger', 'street', 'streets', 'substance', 'sudden', 'summer', 'sun', 'sunset', 'surviving', 'sweet', 'symphonie', 'takes', 'tale', 'tales', 'talk', 'talking', 'talks', 'target', 'texas', 'that', 'theory', 'there', 'thief', 'thieves', 'thin', 'thing', 'things', 'three', 'tie', 'time', 'tin', 'tom', 'top', 'touch', 'town', 'treasure', 'trial', 'trouble', 'true', 'truth', 'turning', 'twist', 'two', 'ulee', 'under', 'unforgettable', 'upon', 'vegas', 'venice', 'very', 'vie', 'vous', 'waiting', 'walk', 'walked', 'walking', 'war', 'warhol', 'warriors', 'was', 'washington', 'water', 'way', 'wedding', 'weekend', 'welcome', 'were', 'west', 'what', 'when', 'while', 'white', 'who', 'wife', 'wild', 'willy', 'window', 'wings', 'with', 'without', 'witness', 'woman', 'women', 'wonderful', 'wonderland', 'wood', 'world', 'wrong', 'year', 'york', 'you', 'young']
        # vectorizer = TfidfVectorizer(min_df=2, max_df=0.25, lowercase=True,  stop_words=stop_words, analyzer='word', token_pattern=r'(?u)\b(?![\d_])\w{3,}\b', ngram_range=(1,2))
        # vectorizer.fit(self.item_features['title'])

        # item_titles = self.item_features['title'].loc[item_ids]
        # item_names_bow = vectorizer.transform(item_titles).toarray()

        # print(vectorizer.get_feature_names_out())

        # item_feats = ag_np.concatenate([item_feats, item_names_bow], axis=1)

        combined_features = ag_np.concatenate([latent_vectors_predictions, user_feats, item_feats], axis=1)
        return combined_features
    




    def train_classifier(self, features, labels):
        # valid auroc ~ 0.76
        # self.classifier = RandomForestClassifier()
        # param_grid = {
        # 'n_estimators': [10, 100, 200],  
        # 'max_features': ['auto', 'sqrt', 'log2'],  
        # 'max_depth': [10, 20, 30, None],  
        # 'min_samples_split': [2, 5, 10],  
        # 'min_samples_leaf': [1, 2, 4], 
        # 'bootstrap': [True, False] 
        # }


        param_grid = {
            'n_estimators': [200],
            'learning_rate': [0.1],
            'max_depth': [5],
            'subsample': [1.0]
        }
        self.classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, subsample=1.0)



        # valid auroc ~ 0.74 
        # self.classifier = SVC(kernel='linear', probability= True)

        # valid auroc ~ 0.64
        # self.classifier = KNeighborsClassifier(n_neighbors=100)

        #valid auroc ~ 0.759
        # self.classifier = LogisticRegression()
        # param_grid = {
        #     'C': [0.1, 1, 10],
        #     'penalty': ['l1', 'l2'],
        #     'solver': ['liblinear', 'saga']
        # }

        # grid_search = GridSearchCV(self.classifier, param_grid=param_grid,
        #                            scoring='roc_auc', cv=5, verbose=1)
        # grid_search.fit(features, labels)
        # self.classifier = grid_search.best_estimator_
        # best_params = grid_search.best_params_
        # best_score = grid_search.best_score_
        # print("Best hyperparameters:", best_params)
        # print("Best cross-validation AUC score:", best_score)
        

        self.classifier.fit(features, labels)




    def predict_classification(self, features):
        return self.classifier.predict_proba(features)[:, 1]  # Probability of class 1




    def evaluate_model(self, features, true_labels):
        predictions = self.predict_classification(features)
        return roc_auc_score(true_labels, predictions)
    


    
if __name__ == '__main__':
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()
    user_features, item_features = load_features()
    
    # Convert ratings to binary labels for training
    train_labels = (train_tuple[2] >= 4.5).astype(int)  # Assuming the third element in the tuple are the ratings

    K = 50

    # Instantiate the recommender system
    recommender = HybridRecommenderSystem(user_features, item_features, step_size=0.1, n_epochs=10, batch_size=100, n_factors=K, alpha=0.01)
    recommender.init_parameter_dict(n_users, n_items, train_tuple)

    recommender.fit(train_tuple, valid_tuple)

    # Prepare features
    train_features = recommender.prepare_features(train_tuple[0], train_tuple[1])

    # Train the classifier
    recommender.train_classifier(train_features, train_labels)

    # Evaluate model (on validation set, for example)
    valid_labels = (valid_tuple[2] >= 4.5).astype(int)
    valid_features = recommender.prepare_features(valid_tuple[0], valid_tuple[1])
    auc_score = recommender.evaluate_model(valid_features, valid_labels)
    print(f"Validation AUC Score for K = {K}:", auc_score)
    print("\n")
    print(f"Validation accuracy Score for K = {K}:", auc_score)



    data = pd.read_csv('..\\data_movie_lens_100k\\ratings_masked_leaderboard_set.csv', usecols=['user_id', 'item_id'])

    # Create a tuple from the user_id and item_id columns
    leaderboard_tuple = (data['user_id'].values, data['item_id'].values)
    leaderboard_features = recommender.prepare_features(leaderboard_tuple[0], leaderboard_tuple[1])
    leaderboard_predictions = recommender.predict_classification(leaderboard_features)

    with open('predicted_ratings_leaderboard.txt', 'w') as file:
        for prediction in leaderboard_predictions:
            file.write(f'{prediction}\n')

