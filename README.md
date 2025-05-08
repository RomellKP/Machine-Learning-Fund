# Projects Machine Learning Fundementals:

---

## Project 1:   

  ### Description:  
  This project implements three types of **Gaussian Discriminant Analysis (GDA)** classifiers in Python using NumPy. These classifiers fall into two classes based on multivariate Gaussian distributions, using different assumptions about the covariance structure.
  

  ### Features:  
  - Implements three GDA classifiers:
    1. **C1**: Class-specific full covariance matrices
    2. **C2**: Shared full covariance matrix
    3. **C3**: Shared diagonal covariance matrix
   - Reads training and test datsets from '.txt' files
   - Computes precision and recall for model evaluation

  ### Concepts used:
  - Multivariate Gaussian Distributions
  - Maximum Liklihood Estimation
  - Discriminant Funtions
  - Covariance Matrix Analysis
  - Performance Metrics (Precision, Recall) 
  
  ### How to Run:
  1. Compile all and run hw1.py

  
  ### My Contributions: 
  - MyDiscriminant.py


## Project 2: Clustering and Dimension Reduction for Handritten Digit Classification

  ### Description:
  This project implements a pipeline combining K-Means Clustering and Principal Cp,[pmemy Analysis (PCA) for classification of handwritten digit images (0, 8 , and 9) from digits089.csv


  ### Features:
  - Implementation of K-means algorithm
  - Implementation of PCA
  - Classification based on majority voting within clusters
  - Evaluation of clustering performance on:
    1. High dimensional data
    2. PCA-reduced data with >95% variance
    3. PCA-reduced data to a single dimension
  - Visualization of reconstruction error over iterations
  
  
  ### How to Run:
  1. Compile all and run hw2.py
  
  ### My Contributions:
  - Mykmeans.py
  - MPCA.py

## Project 3: MLP for Handwritten Digit Classification

### Description: 
This project implements a Multilayer Perceptron (MLP) to classify handwritten digits from the Optical Digits dataset (handwritten digits processed as vectors).
The MLP has one hidden layer with a configurable amount of neurons and is trained using gradient descent.

### Features:
- Normalization of labels
- Fully implemented MLP supporting forward and back propogation
- Evaluation of model performance for different hidden layer sizes to find optimal number of neurons

### Concepts: 
- MLP (neural networks)
- Activation functions
- Backpropagation
- Gradient descent

### How to Run: 
1. Compile all and run hw3.py

### My Contributions: 
- MyMLP.py
  

## Project 4: Decision Tree for Handwritten Digit Classification

  ### Description:
  This project implements a decision tree classifier in Python using NumPy. It uses binary features and allows
  either entropy or the Gini index as the splitting criteria. The classifier is tested on the Optical Digits
  dataset (handwritten digits processed as vectors) to evaluate its accuracy with different values for minimum entropy
  

 ### Features: 
 - Decision Tree implementation
 - Entropy and Gini index impurity measures
 - prediction and evaluation on training, validation, and test sets

  ### Concepts used: 
  - Decision Trees
  - Binary Feature Splits
  - Entropy
  - Gini Index
  
  ### How to Run:
  1. Compile all and run hw4.py
  
  
  ### My Contributions:
  - MyDecisionTree.py
