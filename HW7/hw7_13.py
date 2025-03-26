
# read file

import numpy as np
import matplotlib.pyplot as plt
import sys

# read file and split data into x and y
def read_file(file_name):
    with open(file_name, 'r') as file:
        data = file.readlines()
    x = []
    y = []
    for line in data:
        line = line.strip().split()
        x.append([float(i) for i in line[:-1]])
        y.append(int(line[-1]))
    return np.array(x), np.array(y)



# node consists of s, theta,feature, left,right, isleaf
# if it is leaf node, s, theta, feature are None
class Node:
    def __init__(self, s=None, theta=None, feature=None, left=None, right=None): 
        self.s = s
        self.theta = theta
        self.feature = feature
        self.left = left
        self.right = right
    def predict(self, x):
        if self.s == None:
            return self.theta
        if(self.s * (x[self.feature] - self.theta) < 0):
            return self.left.predict(x)
        return self.right.predict(x)

class Tree:
    def __init__(self):  
        self.root = Node()
    # build tree
    # recursively produce node until the node is leaf
    # iterating all the features,s,theta to find the best split
    # best split is the one with the smallest gini index
    # split the data into two parts, left and right
    def build_tree(self, X, Y):        
       #if all the labels are the same, return the leaf node
        if len(set(Y)) == 1:
            pred = Y[0]
            return Node(s = None, theta = pred, feature = None, left = None, right = None)
        # find the best split
        else:
            s, theta, feature, gini = find_best_split(X, Y)
            left_X, left_Y, right_X, right_Y = split(X, Y, s, theta, feature)
            left = self.build_tree(left_X, left_Y)
            right = self.build_tree(right_X, right_Y)
            return Node(s, theta, feature, left, right)
    # return the number of internal nodes
    def max_branch_depth(self, node):
    # ignore the leaf node
        if node is None or node.s is None:
            return 0  
        left_depth = self.max_branch_depth(node.left)
        right_depth = self.max_branch_depth(node.right)
    
        return 1 + max(left_depth, right_depth)
        
    
    # predict
    def predict(self, x):
        return self.root.predict(x)

    def predict_all(self, X):
        return [self.predict(x) for x in X]



# find the best split
# iterate all the features, s, theta to calculate gini index
# return the minimum gini index and the corresponding

def find_best_split(X, Y):
    s_list = [-1, 1]  
    min_gini = np.inf
    best_s, best_theta, best_feature = None, None, None

    # iterate all the features, s, theta to calculate gini index
    for s in s_list:
        # iterate all the features
        for feature in range(X.shape[1]): 
            theta_values = (X[1:, feature] + X[:-1, feature]) / 2  
            # iterate all the theta values
            for t in theta_values:  
                gini = gini_index(s, t, feature, X, Y)  

                if gini < min_gini:  # update the best split
                    min_gini = gini
                    best_s = s
                    best_theta = t
                    best_feature = feature

    return best_s, best_theta, best_feature, min_gini  # return the best split


# calculate gini index
# formula : 1 - sum(p^2)
# p is the probability of each class
# weighted sum of two parts of gini index 

def gini_index(s, theta, feature, X, Y):

    left_Y = []  
    right_Y = [] 
    
    # split the data into two parts
    for i in range(len(X)):
        if s * (X[i][feature] - theta) < 0:
            left_Y.append(Y[i]) 
        else:
            right_Y.append(Y[i]) 
    left_size = len(left_Y)
    right_size = len(right_Y)
    total_size = len(Y)  

    if left_size == 0 or right_size == 0:
        return 0  # This split is meaningless, return Gini Index = 0

    p1_left = sum(i == 1 for i in left_Y) / left_size  # Probability of class 1 in left partition
    p2_left = sum(i == -1 for i in left_Y) / left_size  # Probability of class -1 in left partition
    left_gini = 1 - p1_left ** 2 - p2_left ** 2  # Compute left Gini impurity

    p1_right = sum(i == 1 for i in right_Y) / right_size  # Probability of class 1 in right partition
    p2_right = sum(i == -1 for i in right_Y) / right_size  # Probability of class -1 in right partition
    right_gini = 1 - p1_right ** 2 - p2_right ** 2  # Compute right Gini impurity

    # Compute the weighted Gini Index
    return (left_size / total_size) * left_gini + (right_size / total_size) * right_gini


# based on the best split, split the data into two parts
def split(X, Y, s, theta, feature):
    left_X = []
    left_Y = []
    right_X = []
    right_Y = []

    for i in range(len(X)):
        if s * (X[i][feature] - theta) < 0:
            left_X.append(X[i])
            left_Y.append(Y[i])
        else:
            right_X.append(X[i])
            right_Y.append(Y[i])

    return np.array(left_X), np.array(left_Y), np.array(right_X), np.array(right_Y)


# main function
# 1. read file
# 2. build CaRT model
# 3. predict
# 4. print


if __name__ == "__main__":

    # read train and test data
    train_X, train_Y = read_file('train.dat')
    test_X, test_Y = read_file('test.dat')

    # build CaRT model
    model = Tree()
    model.build_tree(train_X, train_Y)

        # calculate the number of nodes
    print("Number of nodes: ", model.max_branch_depth(model.root))


    # predict ein and eout
    pred_train = model.predict_all(train_X)
    pred_test = model.predict_all(test_X)

    # calculate the error rate
    ein = sum(pred_train != train_Y) / len(train_Y)
    eout = sum(pred_test != test_Y) / len(test_Y)

    # print
    print("Ein: ", ein)
    print("Eout: ", eout)