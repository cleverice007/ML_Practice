import numpy as np
import math

def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append([float(x) for x in line.split()])
        data = np.array(data)
        X = data[:, :-1]
        Y = data[:, -1]
        X = np.insert(X, 0, 1, axis=1)
    return X , Y


# split training data into 5 folds
# return the list of X and Y of 5 folds

def split_train_data(X, Y):
    X_folds = np.array_split(X, 5)
    Y_folds = np.array_split(Y, 5)
    return X_folds, Y_folds

# implement regularized linear regression
# formula of w = (X^T * X + lambda * I)^-1 * X^T * Y

def regularized_linear_regression(X, Y, lambda_ = 10):
    # Calculate the weight vector
    lambda_I = lambda_ * np.identity(X.shape[1])
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lambda_I), X.T), Y)
    return w

# calculate error rate by using 1/0 error

def calculate_error(X, Y, w):
    
    predictions = np.sign(np.dot(X, w))
    
    errors = np.sum(predictions != Y)
    
    error_rate = errors / len(Y)
    
    return error_rate

# cross validation to calculate ein
# return the average ein

def cross_validation(X_folds, Y_folds, lambda_):
    avg_ecv = 0
    for i in range(5):
        X_train = np.concatenate([X_folds[j] for j in range(5) if j != i])
        Y_train = np.concatenate([Y_folds[j] for j in range(5) if j != i])
        X_validation = X_folds[i]
        Y_validation = Y_folds[i]
        w = regularized_linear_regression(X_train, Y_train, lambda_)
        avg_ecv += calculate_error(X_validation, Y_validation, w)
    avg_ecv /= 5
    return avg_ecv,lambda_

# choose the best lambda by calculating Ecv
# using cross validation function
# return the best lambda, Ecv

def choose_lambda(X_folds, Y_folds, lambda_):
    best_lambda = 0
    best_ecv = float('inf')
    for l in lambda_:
        ecv, _ = cross_validation(X_folds, Y_folds, l)
        if ecv < best_ecv:
            best_ecv = ecv
            best_lambda = l
    return best_lambda, best_ecv


if __name__ == '__main__':
    train_file = "hw4_train.dat"
    X, Y = read_file(train_file)
    X_folds, Y_folds = split_train_data(X, Y)
    lambda_ = [10**i for i in range(2, -11, -1)]
    best_lambda, best_ecv = choose_lambda(X_folds, Y_folds, lambda_)
    print('Best lambda:', best_lambda)
    print('Ecv:', best_ecv)

    # q20 : using optimal lambda to calculate Ein,eout for the whole training data
    # and the test data
    w = regularized_linear_regression(X, Y, best_lambda)
    test_file = "hw4_test.dat"
    X_test, Y_test = read_file(test_file)
    ein = calculate_error(X, Y, w)
    eout = calculate_error(X_test, Y_test, w)
    print('Ein:', ein)
    print('Eout:', eout)