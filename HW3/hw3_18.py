# read hw3_test.dat & hw3_train.dat
# split data into labels and X

import numpy as np

def read_file(file_path):
    """
    Reads a dataset file and splits it into features (X) and labels (Y).

    Parameters:
        file_path (str): Path to the data file.

    Returns:
        X (numpy.ndarray): Feature matrix.
        Y (numpy.ndarray): Label vector.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line by whitespace and convert to float
            values = list(map(float, line.split()))
            data.append(values)
    
    # Convert to numpy array
    data = np.array(data)
    
    # The last column is Y (labels), the rest are X (features)
    X = data[:, :-1]
    Y = data[:, -1]
    
    return X, Y


#  implement logistic regression and update weights using gradient descent
# learning rate = 0.001, number of iterations = 2000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_decent(X, Y, lr=0.001, num_iter=2000):
    # initialize weights
    w = np.zeros(X.shape[1])
    for i in range(num_iter):
        # calculate gradient
        gradient = np.zeros(X.shape[1])
        for x, y in zip(X, Y):
            gradient += y * x * sigmoid(-y * np.dot(w, x))
        gradient /= X.shape[0]
        # update weights
        w += lr * gradient
    return w


# calculate eout using the test data
# calculate mean 0/1 error



# 
if __name__ == "__main__":
    train_file = "hw3_train.dat"
    test_file = "hw3_test.dat"
    
    X_train, Y_train = read_file(train_file)
    X_test, Y_test = read_file(test_file)
    print(X_train.shape, Y_train.shape)
    print(Y_train[:5])
   


