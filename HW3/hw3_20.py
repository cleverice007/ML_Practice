#read file ,split data into X and Y
# add a column of 1 to X

import numpy as np

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
    return X, Y

# learning rate = 0.001, number of iterations = 2000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# update weights using stochastic gradient descent
# update weights one sample at a time
# formula of gradient: (sigmoid(w^T * x) - y) * x

def update_weights(X, Y, learning_rate=0.001, num_iterations=2000):
    # Initialize weights to zeros
    w = np.zeros(X.shape[1])
    size = X.shape[0]  # Number of samples

    # Perform gradient descent
    for _ in range(num_iterations):
        nabla_err = np.zeros(X.shape[1])  # Initialize gradient accumulator to zeros

        # Compute gradient by iterating over all samples
        for i in range(size):
            val1 = np.dot(X[i], w)  
            val2 = -1 * Y[i] * val1  
            val3 = 1 / (1 + np.exp(-1 * val2))  
            val = val3 * (-1) * Y[i] * X[i]  
            nabla_err = nabla_err + val 

        nabla_Ein = nabla_err / size  
        w = w - learning_rate * nabla_Ein 

    return w
# calculate mean 0/1 error

def calculate_error(X, Y, w):
    
    predictions = np.sign(np.dot(X, w))
    
    errors = np.sum(predictions != Y)
    
    error_rate = errors / len(Y)
    
    return error_rate

if __name__ == "__main__":
    train_file = "hw3_train.dat"
    test_file = "hw3_test.dat"
    
    X_train, Y_train = read_file(train_file)
    X_test, Y_test = read_file(test_file)
    w = update_weights(X_train, Y_train)
    error_rate = calculate_error(X_test, Y_test, w)
    print("Weights: ", w)
    print("Error rate: ", error_rate)