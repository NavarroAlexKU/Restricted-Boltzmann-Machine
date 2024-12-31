# Restricted Boltzmann Machine (RBM) for Movie Recommendation
![image](https://github.com/user-attachments/assets/fd8376ae-8d4e-4939-8be9-9e2d82e1a0d0)

This project implements a **Restricted Boltzmann Machine (RBM)** for collaborative filtering, specifically to build a movie recommendation system. The RBM learns latent features of users and movies from the MovieLens dataset and predicts whether users will like unseen movies.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [RBM Overview](#rbm-review)
5. [Insights](#insights)

---

## Project Overview

The goal of this project is to implement an RBM to:

- Learn latent features from user-movie interaction data.
- Predict user preferences for movies they have not rated yet.

An RBM is a type of generative stochastic neural network that models the joint probability distribution of visible and hidden layers. In this project:

- **Visible Nodes**: Represent user ratings for movies.
- **Hidden Nodes**: Represent latent features of users and movies.

---

## RBM Review

### Visible Nodes:
- Represent the **input layer** of the RBM.
- Correspond to the **observed data** (e.g., user ratings or interactions).
- Each node corresponds to a feature or variable in the data (e.g., a specific movie or product and whether a user has interacted with it, such as rating or purchasing it).

### Hidden Nodes:
- Represent the **latent (hidden) feature layer** of the RBM.
- Capture patterns, relationships, or structures in the data that are **not directly observable**.
- Help the RBM learn **abstract features** from the visible data.
  - Example: A hidden node might detect that a user prefers a certain movie genre (e.g., action over comedy) or likes specific themes in products.

### Lines in the RBM:
- Represent the **weights** between visible and hidden nodes.
- **Weights** are adjusted (learned) through a combination of:
  1. **Energy-Based Models**:
     - Assign probabilities to different configurations of visible and hidden nodes using an **energy function**.
  2. **Contrastive Divergence (CD)**:
     - An algorithm used to train the RBM by minimizing the difference between the observed and reconstructed data.

## Dataset

This project uses the **MovieLens** dataset, which consists of user ratings for movies.

### Files:
- `movies.dat`: Information about movies (ID, title, genres).
- `users.dat`: User metadata (ID, gender, age, occupation, zip code).
- `ratings.dat`: User-movie interaction data (user ID, movie ID, rating).
- `u1.base`: Training set.
- `u1.test`: Test set.

---

## Model Architecture

The RBM consists of:

1. **Visible Layer**:
   - Nodes correspond to movies.
   - Inputs are user ratings for movies.

2. **Hidden Layer**:
   - Nodes represent latent features extracted from user-movie interactions.

3. **Weight Matrix (W)**:
   - Encodes connections between visible and hidden layers.

4. **Bias Terms (a, b)**:
   - Control the activation probabilities of the hidden and visible layers.

Training is performed using **Contrastive Divergence (CD)**, which involves multiple steps of Gibbs sampling.

---

## Insights
The model's performance on reconstruction can be evaluated by comparing the original ratings to the reconstructed ratings. In cases where the model had sufficient data to learn user preferences, it was able to accurately replicate the original ratings. This indicates that the RBM successfully captured patterns in the data for those scenarios.

Below is a visualization comparing the original and reconstructed ratings for a sample user:
![image](https://github.com/user-attachments/assets/72974a47-ac4c-419e-a526-3328c92aff2b)

Let's See the Model's Movie Recommendations for a Random User
```
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
```

![image](https://github.com/user-attachments/assets/3ed51b5b-c8a3-4df6-8bef-de50a1b4633e)

