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


# split train dataset into 2 parts
# 120 for training, 80 for validation

def split_train_validation(X, Y):
    X_train = X[:120]
    Y_train = Y[:120]
    X_validation = X[120:]
    Y_validation = Y[120:]
    return X_train, Y_train, X_validation, Y_validation

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
# choose the best lambda by calculating Ein
# return the lambda ,Ein ,Eval, Eout

def choose_lambda(X_train, Y_train, X_validation, Y_validation, lambda_):
    best_lambda = 0
    best_ein = float('inf')
    best_eval = float('inf')
    best_eout = float('inf')
    for l in lambda_:
        w = regularized_linear_regression(X_train, Y_train, l)
        ein = calculate_error(X_train, Y_train, w)
        eval_ = calculate_error(X_validation, Y_validation, w)
        eout = calculate_error(X_test, Y_test, w)
        if ein < best_ein:
            best_ein = ein
            best_lambda = l
            best_eval = eval_
            best_eout = eout
    return best_lambda, best_ein, best_eval, best_eout

if __name__ == '__main__':
    train_file = "hw4_train.dat"
    test_file = "hw4_test.dat"
    X_train, Y_train = read_file(train_file)
    X_test, Y_test = read_file(test_file)
    X_train, Y_train, X_validation, Y_validation = split_train_validation(X_train, Y_train)
    X_test, Y_test = read_file('hw4_test.dat')
    lambda_ = [10**i for i in range(2, -11, -1)]
    best_lambda, best_ein, best_eval, best_eout = choose_lambda(X_train, Y_train, X_validation, Y_validation, lambda_)
    print('Best lambda:', best_lambda)
    print('Ein:', best_ein)
    print('Eval:', best_eval)
    print('Eout:', best_eout)