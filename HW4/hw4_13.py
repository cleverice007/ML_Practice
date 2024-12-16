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
    return X , Y

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


# calculate ein and eout
if __name__ == "__main__":

    train_file = "hw4_train.dat"
    test_file = "hw4_test.dat"
    
    X_train, Y_train = read_file(train_file)
    X_test, Y_test = read_file(test_file)

    w_in = regularized_linear_regression(X_train, Y_train)
    w_out = regularized_linear_regression(X_train, Y_train, 10)

    ein = calculate_error(X_train, Y_train, w_in)
    eout = calculate_error(X_test, Y_test, w_out)
    
    print("Ein: ", ein)
    print("Eout: ", eout)