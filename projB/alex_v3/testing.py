import numpy as np
import matplotlib.pyplot as plt
from train_valid_test_loader import load_train_valid_test_datasets
import CollabFilterOneVectorPerItem as collabf
import pandas as pd

# Load the dataset
train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()


'''
------------------------------------------
1a
------------------------------------------
'''


# parameters
factors = [2, 10, 50]
alpha = 0  # Alpha set to 0 for part 1a
n_epochs = 1000
batch_size = 32
step_size = 0.2
mae_results = {}
models = []

# Train a model for each factor and compute the MAE data
'''
for factor in factors:
    model = collabf.CollabFilterOneVectorPerItem(
        n_epochs=n_epochs, batch_size=batch_size, step_size=step_size,
        n_factors=factor, alpha=alpha)
    model.init_parameter_dict(n_users, n_items, train_tuple)
    model.fit(train_tuple, valid_tuple)
    mae_results[factor] = model.trace_mae_train
    models.append(model)

# Print results at the end of training for each model
for i, model in enumerate(models):
    print(f'At End: Model for K={factors[i]}, Train AUC: {model.trace_auc_train[-1]}, '
          f'Valid AUC: {model.trace_auc_valid[-1]}, Train MAE: {model.trace_mae_train[-1]}, '
          f'Valid MAE: {model.trace_mae_valid[-1]}')

#---------------1a plot------------------------------
idx = 1
for factor in factors:
    plt.subplot(1, 3, idx)
    plt.plot(mae_results[factor], label=f'K={factor}')
    plt.title(f'K={factor}')
    plt.xlabel('Epoch')
    plt.ylabel('Train MAE')
    plt.legend()
    idx += 1
plt.tight_layout()
plt.show()
'''
'''
------------------------------------------
1b
------------------------------------------
'''
'''
n_epochs = 50
alphas = [0.001, 0.01, 0.1]  # Array of alpha values to test
factors_1b = 50
valid_mae_results = {}

for alpha in alphas:
    model_1b = collabf.CollabFilterOneVectorPerItem(
        n_epochs=n_epochs, batch_size=batch_size, step_size=step_size,
        n_factors=factors_1b, alpha=alpha)
    model_1b.init_parameter_dict(n_users, n_items, train_tuple)
    model_1b.fit(train_tuple, valid_tuple)
    valid_mae_results[alpha] = model_1b.trace_mae_valid

# Plot results for part 1b
plt.figure(figsize=(8, 6))
for alpha, auc in valid_mae_results.items():
    plt.plot(auc, label=f'Alpha={alpha}')
plt.title('Trace Plot for K=50 with Alpha > 0')
plt.xlabel('Epoch')
plt.ylabel('Valid MAE')
plt.legend()
plt.show()
'''

'''
------------------------------------------
1c
------------------------------------------
'''

# Best Params
factor = 2
alpha = 0.1
n_epochs = 600
batch_size = 32
step_size = 0.2

model = collabf.CollabFilterOneVectorPerItem(
    n_epochs=n_epochs, batch_size=batch_size, step_size=step_size,
    n_factors=factor, alpha=alpha)
model.init_parameter_dict(n_users, n_items, train_tuple)
model.fit(train_tuple, valid_tuple)

V = model.param_dict['V']
if V.shape == (2, 1682):
    V = V.T

embeddings_df = pd.DataFrame(V, columns=['x', 'y'])
movie_info_df = pd.read_csv('C:\\Users\\alexc\\CS135\\projB-release\\data_movie_lens_100k\\select_movies.csv')
plot_data = pd.merge(embeddings_df, movie_info_df, left_index=True, right_on='item_id')

plt.figure()

# Iterate through each movie in the dataset
for index, movie in plot_data.iterrows():
    # Extract the x and y coordinates
    x_coordinate = movie['x']
    y_coordinate = movie['y']
    movie_title = movie['title']
    plt.scatter(x_coordinate, y_coordinate, color='blue', marker='.')
    
    # Label each point with the movie title
    plt.text(x_coordinate, y_coordinate, movie_title, fontsize=9, ha='right')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Embedding Visualization for K=2')
plt.grid(True)
plt.show()

