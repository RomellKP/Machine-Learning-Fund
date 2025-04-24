import numpy as np
# some tips
# |S| is the determinant of S in the discriminant functions, try np.linalg.det()
# you can also directly get the inverse of a matrix by np.linalg.inv()


# ------------------------------------- You are going to implement 3 classifiers and corresponding helper functions --------------------
# ------------------------------------- Three classifiers start from here --------------------------------------------------------------
class GaussianDiscriminantBase:
    def __init__(self) -> None:
        pass

    def calculate_metrics(self, ytest, predictions):
        precision = compute_precision(ytest, predictions)
        recall = compute_recall(ytest, predictions)
        return precision, recall

class GaussianDiscriminant_C1(GaussianDiscriminantBase):
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))   # S1 and S2, store in 2*(8*8) matrices
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        ## filling in your code here !!!!!!!!!!!!!!!!! add a line to compute S1 for class 1
        self.S[0,:,:] = computeCov(Xtrain1)
        
        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        ## filling in your code here !!!!!!!!!!!!!!!!! add a line to compute S2 for class 2
        self.S[1,:,:] = computeCov(Xtrain2)
        
        
        # priors for both class
        self.p = computePrior(ytrain)

    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        # can be ignored, removed or replaced with any following implementations
        #predictions = np.zeros(Xtest.shape[0])

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finall get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)
        g1 = self.discFun(Xtest, 0)
        g2 = self.discFun(Xtest, 1)

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        predictions = np.where(g1 > g2, 1, 2)

        return predictions
    
    def discFun(self, X, class_idx):
        diff = X - self.m[class_idx]
        sd = np.sqrt(np.diag(self.S[class_idx]))
        logSd = np.log(sd)
        diffSqr = np.square(diff) / (2 * np.square(sd))
        return -0.5 * np.sum(np.log(2 * np.pi)) - np.sum(logSd) - np.sum(diffSqr, axis=1) + np.log(self.p[class_idx])

class GaussianDiscriminant_C2(GaussianDiscriminantBase):
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.shared_S =np.zeros((d,d))  # the shared convariance S that will be used for both classes
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        # m2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step 3: Compute the shared covariance matrix that is used for both class
        # shared_S is computed by finding a covariance matrix of all the data
        self.shared_S = np.cov(Xtrain.T) 

    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        # can be ignored, removed or replaced with any following implementations
        predictions = np.zeros(Xtest.shape[0])

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finall get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)
        g1 = self.discFun(Xtest, 0)
        g2 = self.discFun(Xtest, 1)

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        predictions = np.where(g1 > g2, 1, 2)

        return np.array(predictions)

    def discFun(self, X, class_idx):
        diff = X - self.m[class_idx]
        sd = np.sqrt(np.diag(self.shared_S))
        logSd = np.log(sd)
        diffSqr = np.square(diff) / (2 * np.square(sd))
        return -0.5 * np.sum(np.log(2 * np.pi)) - np.sum(logSd) - np.sum(diffSqr, axis=1) + np.log(self.p[class_idx])

class GaussianDiscriminant_C3(GaussianDiscriminantBase):
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.shared_S =np.zeros((d,d))  # the shared convariance S that will be used for both classes
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        # m2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step 3: Compute the shared covariance matrix that is used for both class
        # shared_S is computed by finding a covariance matrix of all the data 
        self.shared_S = np.cov(Xtrain.T)
        
        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step 4: Compute the diagonal of shared_S
        # [[1,2],[2,4]] => [[1,0],[0,4]], try np.diag()
        self.shared_S = np.diag(np.diag(self.shared_S))

    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        # can be ignored, removed or replaced with any following implementations
        predictions = np.zeros(Xtest.shape[0])

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finall get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)
        # Please note here, currently we assume shared_S is a d*d diagonal matrix, the non-capital si^2 in the lecture formula will be the i-th entry on the diagonal
        g1 = self.discFun(Xtest, 0)
        g2 = self.discFun(Xtest, 1)

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        predictions = np.where(g1 > g2, 1, 2)
        return np.array(predictions)
    
    def discFun(self, X, class_idx):
        diff = X - self.m[class_idx]
        sd = np.sqrt(np.diag(self.shared_S))
        logSd = np.log(sd)
        diffSqr = np.square(diff) / (2 * np.square(sd))
        return -0.5 * np.sum(np.log(2 * np.pi)) - np.sum(logSd) - np.sum(diffSqr, axis=1) + np.log(self.p[class_idx])



# ------------------------------------- Helper Functions start from here --------------------------------------------------------------
# Input:
# features: n*d matrix (n is the number of samples, d is the number of dimensions of the feature)
# labels: n vector
# Output:
# features1: n1*d
# features2: n2*d
# n1+n2 = n, n1 is the number of class1, n2 is the number of samples from class 2
def splitData(features, labels):
    # placeholders to store the separated features (feature1, feature2), 
    # can be ignored, removed or replaced with any following implementations
    #features1 = np.zeros([np.sum(labels == 1),features.shape[1]])  
    #features2 = np.zeros([np.sum(labels == 2),features.shape[1]])

    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!!
    # separate the features according to the corresponding labels, for example
    # if features = [[1,1],[2,2],[3,3],[4,4]] and labels = [1,1,1,2], the resulting feature1 and feature2 will be
    # feature1 = [[1,1],[2,2],[3,3]], feature2 = [[4,4]]
    features1 = features[labels == 1]
    features2 = features[labels == 2]

    return features1, features2


# compute the mean of input features
# input: 
# features: n*d
# output: d
def computeMean(features):
    # placeholders to store the mean for one class
    # can be ignored, removed or replaced with any following implementations
    #m = np.zeros(features.shape[1])
    m = np.mean(features, axis=0)

    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!! 
    # try to explore np.mean() for convenience
    return m


# compute the mean of input features
# input: 
# features: n*d
# output: d*d
def computeCov(features):
    # placeholders to store the covariance matrix for one class
    # can be ignored, removed or replaced with any following implementations
    #S = np.eye(features.shape[1])

    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!!
    # try to explore np.cov() for convenience
    S = np.cov(features.T)
    return S


# compute the priors of input features
# input: 
# labels: n*1
# output: 2
def computePrior(labels):
    # placeholders to store the priors for both class
    # can be ignored, removed or replaced with any following implementations
    #p = np.array([0.5,0.5])

    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!! 
    # p1 = numOf class1 / numOf all the data; same as p2
    p1 = np.sum(labels == 1) / len(labels)
    p2 = np.sum(labels == 2) / len(labels)
    return np.array([p1, p2])


# compute the precision
# input:
# ytest: the ground truth labels of the test data, n*1
# predictions: the predicted labels of the test data, n*1
# output:
# precision: a float with size 1
def compute_precision(ytest, predictions):
    #precision = 0.0 # a place holder can be neglected

    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!
    # precision = countOf[true positive predictions] / countOf[positive predictions]
    # here we assume label==2 is the positive label
    truePos = np.sum((ytest == 2) & (predictions == 2))
    posPred = np.sum(predictions == 2)
    precision = truePos / posPred
    return precision

# compute the recall
# input:
# ytest: the ground truth labels of the test data, n*1
# predictions: the predicted labels of the test data, n*1
# output:
# recall: a float with size 1
def compute_recall(ytest, predictions):
    recall = 0.0 # a place holder can be neglected

    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!
    # precision = countOf[true positive predictions] / countOf[positive labels in ytest]
    # here we assume label==2 is the positive label
    truePos = np.sum((ytest == 2) & (predictions == 2))
    posLab = np.sum(ytest == 2)
    recall = truePos / posLab
    return recall 