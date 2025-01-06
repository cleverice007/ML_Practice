

# read the file convert it into X and Y

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score


def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append([float(x) for x in line.split()])
        data = np.array(data)
        X = data[:, 1:]
        Y = data[:, 0]
    return X, Y

# convert Y label to 1 and -1
# if Y =0 convert it to  1
# if Y != 0 convert it to -1

def convert_Y(Y):
    Y = np.where(Y == 0, 1, -1)
    return Y


if __name__ == "__main__":
    # read test and train data
    train_file = 'features.train'
    train_X, train_Y = read_file(train_file)
    train_Y = convert_Y(train_Y)

    
    test_file = 'features.test'
    test_X, test_Y = read_file(test_file)
    test_Y = convert_Y(test_Y)

    # set C = 0.01, use soft margin SVM
    C = 0.01
    clf = svm.LinearSVC(loss='hinge', C=C)
    clf.fit(train_X, train_Y)

    # calculate weight norm
    w = clf.coef_
    w_norm = np.sqrt(np.sum(w**2))
    print(f"Weight Norm (||w||): {w_norm:.4f}")

    # calculate E_in
    train_pred = clf.predict(train_X)
    E_in = 1 - accuracy_score(train_Y, train_pred)
    print(f"E_in (Training Error): {E_in:.4f}")

    # calculate E_out
    test_pred = clf.predict(test_X)
    E_out = 1 - accuracy_score(test_Y, test_pred)
    print(f"E_out (Testing Error): {E_out:.4f}")