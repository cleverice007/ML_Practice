import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

# read file
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
def convert_Y(Y, digit):
    Y = np.where(Y == digit, 1, -1)
    return Y

# SVM with normal kernel
# C = 0.1
# calculate eout

if __name__ == "__main__":
    train_file = 'features.train'
    test_file = 'features.test'
    train_X, train_Y = read_file(train_file)
    test_X, test_Y = read_file(test_file)
    
    C = 0.1
    gamma_list = [0.001,0.01,0.1,1,10]
    E_out = []
    train_Y = convert_Y(train_Y, 0)
    test_Y = convert_Y(test_Y, 0)
    for gamma in gamma_list:
        clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        clf.fit(train_X, train_Y)
        test_pred = clf.predict(test_X)
        E_out.append(1 - accuracy_score(test_Y, test_pred))
    min_E_out = min(E_out)
    min_E_out_gamma = gamma_list[E_out.index(min_E_out)]
    print(f"Min E_out: {min_E_out:.4f} for gamma {min_E_out_gamma}")
    print(E_out)