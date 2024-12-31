#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import python packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score


# In[2]:


# Import datasets
movies = pd.read_csv(r'movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv(r'users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv(r'ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
movies.head()


# In[3]:


ratings.head()


# In[4]:


# Prepare training and test set
training_set = pd.read_csv('u1.base', delimiter = '\t')
test_set = pd.read_csv('u1.test', delimiter = '\t')


# In[5]:


# Convert training & test set to numpy array
training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')


# In[6]:


# Get max number of users and movies
nb_users = max(set(training_set[:, 0]).union(set(test_set[:, 0])))
nb_movies= max(set(training_set[:, 1]).union(set(test_set[:, 1])))
print(f"The dataset contains {nb_users} unique users and {nb_movies} unique movies.")


# In[7]:


def convert(data):
    """
    Converts the raw dataset into a matrix with users as rows and movies as columns.
    
    Each user's row contains their ratings for the corresponding movies, with zeros 
    for movies they haven't rated.

    Parameters:
        data (numpy.ndarray): A dataset where each row is [user_id, movie_id, rating].

    Returns:
        list: A 2D list where each row corresponds to a user, and each column corresponds 
              to a movie. Entries are ratings (or 0 if the user hasn't rated the movie).
    """
    # Initialize an empty list to store the converted data
    new_data = []

    # Loop through all user IDs from 1 to the maximum number of users (inclusive)
    for id_users in range(1, nb_users + 1):
        # Extract movie IDs rated by the current user
        id_movies = data[:, 1][data[:, 0] == id_users]
        
        # Extract corresponding ratings for the current user
        id_ratings = data[:, 2][data[:, 0] == id_users]
        
        # Initialize an array of zeros for all movies (default is unrated)
        ratings = np.zeros(nb_movies)
        
        # Replace zeros with actual ratings where the user has rated a movie
        ratings[id_movies - 1] = id_ratings  # Adjust index by subtracting 1
        
        # Append the user's ratings as a list to the new data
        new_data.append(list(ratings))
    
    return new_data


# In[8]:


# Execute function on train and test set
training_set = convert(training_set)
test_set = convert(test_set)


# In[9]:


# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
print(type(training_set))


# In[10]:


# Converting the ratings into binary ratings 1 (Liked) and 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

# Converting the ratings into binary ratings 1 (Liked) and 0 (Not Liked)
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# Creating the architecture of the Neural Network Recommendation System using an RBM involves defining the following components:
# 
# - Number of Hidden Nodes:
#     * Determines the size of the hidden layer, which represents latent features learned by the RBM.
# - Weight Matrix:
#     * Encodes the relationship between visible nodes (e.g., movies) and hidden nodes (e.g., latent factors).
# - Bias Terms:
#     * Visible bias: Controls activation probabilities of the visible nodes.
#      * Hidden bias: Controls activation probabilities of the hidden nodes.
#      
# These parameters allow the RBM to model the joint probability distribution of the visible and hidden layers, enabling learning and reconstruction of data.

# In[11]:


class RBM():
    """
    Restricted Boltzmann Machine (RBM) implementation for collaborative filtering and recommendation systems.

    Attributes:
        W (torch.Tensor): Weight matrix representing connections between visible and hidden layers.
        a (torch.Tensor): Bias vector for the hidden layer, controlling hidden node activations.
        b (torch.Tensor): Bias vector for the visible layer, controlling visible node activations.
    """

    def __init__(self, nv, nh):
        """
        Initializes the RBM with the specified number of visible and hidden nodes.

        Args:
            nv (int): Number of visible nodes (e.g., number of features or movies).
            nh (int): Number of hidden nodes (e.g., latent features for learning patterns).
        """
        # Initialize the weight matrix (nh x nv) with random values for connections
        self.W = torch.randn(nh, nv)
        
        # Initialize the bias vector for the hidden nodes (1 x nh)
        self.a = torch.randn(1, nh)
        
        # Initialize the bias vector for the visible nodes (1 x nv)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        """
        Computes the probabilities of the hidden nodes being activated given the visible nodes (`P(h|v)`).
        This step identifies hidden patterns or latent features from the input data.

        Args:
            x (torch.Tensor): The visible layer input data.

        Returns:
            tuple:
                - p_h_given_v (torch.Tensor): Probabilities of hidden nodes being activated.
                - sample (torch.Tensor): Binary sample of hidden nodes based on the probabilities.
        """
        # Calculate the weighted input to the hidden nodes
        wx = torch.mm(x, self.W.t())  # Matrix multiplication of visible nodes and weight matrix
        activation = wx + self.a.expand_as(wx)  # Add bias for the hidden nodes

        # Apply the sigmoid function to calculate probabilities
        p_h_given_v = torch.sigmoid(activation)

        # Sample binary values (1s and 0s) from the probabilities
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """
        Computes the probabilities of the visible nodes being activated given the hidden nodes (`P(v|h)`).
        This step reconstructs the input data based on the learned patterns from the hidden nodes.

        Args:
            y (torch.Tensor): The hidden layer input data.

        Returns:
            tuple:
                - p_v_given_h (torch.Tensor): Probabilities of visible nodes being activated.
                - sample (torch.Tensor): Binary sample of visible nodes based on the probabilities.
        """
        # Calculate the weighted input to the visible nodes
        wy = torch.mm(y, self.W)  # Matrix multiplication of hidden nodes and weight matrix
        activation = wy + self.b.expand_as(wy)  # Add bias for the visible nodes

        # Apply the sigmoid function to calculate probabilities
        p_v_given_h = torch.sigmoid(activation)

        # Sample binary values (1s and 0s) from the probabilities
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        """
        Updates the weights and biases of the RBM using Contrastive Divergence (CD).

        Contrastive Divergence minimizes the difference between the original data (`v0`) 
        and the reconstructed data (`vk`) by adjusting the weights (`W`) and biases (`a`, `b`).

        Args:
            v0 (torch.Tensor): Original visible layer data.
            vk (torch.Tensor): Reconstructed visible layer after k Gibbs sampling steps.
            ph0 (torch.Tensor): Probabilities of the hidden layer for v0.
            phk (torch.Tensor): Probabilities of the hidden layer for vk.
        """
        # Update the weights based on the difference between original and reconstructed data
        self.W += torch.mm(v0.t(), ph0).t() - torch.mm(vk.t(), phk).t()

        # Update the bias for visible nodes
        self.b += torch.sum((v0 - vk), 0)

        # Update the bias for hidden nodes
        self.a += torch.sum((ph0 - phk), 0)


# Number of visible nodes (e.g., number of features)
nv = len(training_set[0])

# Number of hidden nodes (e.g., latent features for patterns)
nh = 100

# Batch size for training (number of samples per batch)
batch_size = 100

# Initialize the RBM with the specified number of visible and hidden nodes
rbm = RBM(nv, nh)


# In[12]:


# Training the RBM:
# Number of training epochs (full passes over the training dataset)
nb_epoch = 10

# Loop over each epoch
for epoch in range(1, nb_epoch + 1):
    train_loss = 0  # Initialize training loss for this epoch
    s = 0.  # Counter for the number of batches processed

    # Loop through the dataset in batches
    for id_user in range(0, nb_users - batch_size, batch_size):
        # Get a batch of training data for the visible layer
        vk = training_set[id_user:id_user+batch_size]  # Reconstructed visible nodes (starts as original data)
        v0 = training_set[id_user:id_user+batch_size]  # Original visible nodes (for comparison)

        # Compute the probabilities of the hidden nodes given the original visible nodes
        ph0, _ = rbm.sample_h(v0)

        # Perform Contrastive Divergence (CD) with 10 steps of Gibbs sampling
        for k in range(10):
            # Sample hidden nodes given the current visible nodes
            _, hk = rbm.sample_h(vk)
            # Reconstruct visible nodes given the sampled hidden nodes
            _, vk = rbm.sample_v(hk)
            # Ensure missing values (v0 < 0) are not modified during reconstruction
            vk[v0 < 0] = v0[v0 < 0]

        # Compute the probabilities of the hidden nodes after the final step of Gibbs sampling
        phk, _ = rbm.sample_h(vk)

        # Train the RBM by updating weights and biases
        rbm.train(v0, vk, ph0, phk)

        # Calculate the training loss as the mean absolute error between original and reconstructed visible nodes
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))

        # Increment the batch counter
        s += 1.

    # Print the average loss for this epoch
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))


# In[13]:


# Testing the RBM to evaluate its performance on the test set
test_loss = 0  # Initialize the test loss
s = 0.  # Counter for the number of users with valid test data

# Loop through each user in the dataset
for id_user in range(nb_users):
    # Get the visible nodes (input data) for the current user from the training set
    v = training_set[id_user:id_user+1]  # Data from the training set
    
    # Get the visible nodes (input data) for the current user from the test set
    vt = test_set[id_user:id_user+1]  # Data from the test set (used for comparison)
    
    # Proceed only if the test data has valid ratings (entries >= 0)
    if len(vt[vt >= 0]) > 0:
        # Sample the hidden nodes given the visible nodes
        _, h = rbm.sample_h(v)
        
        # Reconstruct the visible nodes given the hidden nodes
        _, v = rbm.sample_v(h)
        
        # Compute the test loss as the mean absolute error for valid ratings
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        
        # Increment the counter for valid test samples
        s += 1.

# Print the average test loss
print('test loss: ' + str(test_loss / s))


# In[14]:


# Select a user for visualization (e.g., user 0)
user_id = 0
original_ratings = test_set[user_id].numpy()
reconstructed_ratings = v[user_id].detach().numpy()

# Get indices of top-N highest-rated movies in the original ratings
N = 10  # Number of movies to display
top_n_indices = original_ratings.argsort()[-N:][::-1]

# Filter original and reconstructed ratings for the top-N movies
original_top_n = original_ratings[top_n_indices]
reconstructed_top_n = reconstructed_ratings[top_n_indices]

# Plot original vs reconstructed ratings for top-N movies
plt.figure(figsize=(10, 5))
plt.bar(range(N), original_top_n, alpha=0.6, label='Original Ratings')
plt.bar(range(N), reconstructed_top_n, alpha=0.6, label='Reconstructed Ratings')
plt.xlabel('Top-N Movie Index')
plt.ylabel('Rating')
plt.title(f'Original vs Reconstructed Ratings for User {user_id} (Top {N} Movies)')
plt.xticks(range(N), labels=top_n_indices, rotation=45)
plt.legend()
plt.show()


# In[21]:


import random

# Randomly select a valid user ID
user_id = random.randint(0, nb_users - 1)  # Randomly pick a user between 0 and nb_users-1

# Get the user's data from the training set
v = training_set[user_id:user_id+1]
_, h = rbm.sample_h(v)  # Sample hidden nodes
_, v_reconstructed = rbm.sample_v(h)  # Reconstruct visible nodes

# Get top-N recommendations
N = 5
recommended = v_reconstructed.numpy().argsort()[0][-N:][::-1]  # Top-N movies with highest predicted ratings
print(f"Top-{N} Recommendations for User {user_id}: {recommended}")

# Retrieve and display recommended movie details
recommended_movies = movies.loc[movies[0].isin(recommended)]
print(recommended_movies)


# In[ ]:





# In[ ]:





# In[ ]:




