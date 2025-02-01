import numpy as np
import sys


# read file and split into X and Y

def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        X = []
        Y = []
        for line in lines:
            data = line.strip().split() # remove '\n' and split by space
            Ｘ.append([float(x) for x in data[:-1]])
            Y.append(int(data[-1]))
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y



# implement decision stump for one feature based on sample with weights

# get the threshold for the feature

def get_threshold(data):
    data = np.sort(data)  
    threshold = (data[:-1] + data[1:]) / 2  
    return threshold

# get the error rate for the feature 
# sample with weights

def get_error_rate(data, label, weight, s, threshold):
    predict = s * np.sign(data - threshold)  
    errors = (predict != label).astype(int)  
    weighted_error = np.dot(weight, errors)  
    return weighted_error

# get the best threshold for the feature

def decision_stump(data, label, weight):
    min_error = float('inf')
    best_s = 1
    best_threshold = 0
    best_feature = -1 

    for s in [-1, 1]:
        for i in range(data.shape[1]):  
            threshold = get_threshold(data[:, i]) 
            for t in threshold:
                error = get_error_rate(data[:, i], label, weight, s, t)
                if error < min_error:
                    min_error = error
                    best_s = s
                    best_threshold = t
                    best_feature = i  

    return min_error, best_s, best_threshold, best_feature


# implement adaboost

def adaboost(X, Y, T=300):
    N = X.shape[0]
    M = X.shape[1]
    weight = np.ones(N) / N
    alpha = np.zeros(T)
    G = np.zeros((T, 3))  
    for t in range(T):
        error, s, threshold, feature = decision_stump(X, Y, weight)
        alpha[t] = 0.5 * np.log((1 - error) / error)
        predict = s * np.sign(X[:, feature] - threshold)
        weight = weight * np.exp(-alpha[t] * Y * predict)
        weight = weight / np.sum(weight)
        G[t] = [feature, s, threshold]
    return G, alpha


    # calculate ein(g1)

if __name__ == '__main__':
    train_file = 'adaboost_train.dat'
    train_X, train_Y = read_file(train_file)
    min_error, best_s, best_threshold, best_feature = decision_stump(train_X, train_Y, np.ones(train_X.shape[0]) / train_X.shape[0])
    print('Ein(g1):', min_error )