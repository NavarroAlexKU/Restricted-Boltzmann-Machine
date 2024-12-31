# Restricted Boltzmann Machine (RBM) for Movie Recommendation
![image](https://github.com/user-attachments/assets/fd8376ae-8d4e-4939-8be9-9e2d82e1a0d0)

This project implements a **Restricted Boltzmann Machine (RBM)** for collaborative filtering, specifically to build a movie recommendation system. The RBM learns latent features of users and movies from the MovieLens dataset and predicts whether users will like unseen movies.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
5. [Usage](#usage)
6. [Results](#results)
7. [References](#references)

---

## Project Overview

The goal of this project is to implement an RBM to:

- Learn latent features from user-movie interaction data.
- Predict user preferences for movies they have not rated yet.

An RBM is a type of generative stochastic neural network that models the joint probability distribution of visible and hidden layers. In this project:

- **Visible Nodes**: Represent user ratings for movies.
- **Hidden Nodes**: Represent latent features of users and movies.

---

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

## Usage

### Training
Run the script to train the RBM:
```bash
python train_rbm.py
