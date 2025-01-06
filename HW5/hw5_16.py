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

# SVM with polynomial kernel
# C = 0.01, Q = 2 coefficient = 1
# calculate E_in
if __name__ == "__main__":
    train_file = 'features.train'
    train_X, train_Y = read_file(train_file)
    
    Q = 2
    C = 0.01

    # 計算不同 digits 的 E_in
    digits = [0, 2, 4, 6, 8]
    E_in = []
    alpha_sum_list = []

    for digit in digits:
        train_Y_digit = convert_Y(train_Y, digit)
        clf = svm.SVC(kernel='poly', C=C, degree=Q, coef0=1, gamma=1)
        clf.fit(train_X, train_Y_digit)
        train_pred = clf.predict(train_X)
        E_in.append(1 - accuracy_score(train_Y_digit, train_pred))
        # find the maximum  sum of support vectors
        alpha_sum = np.sum(np.abs(clf.dual_coef_))
        alpha_sum_list.append(alpha_sum)

    # 找出最小的 E_in 和對應的 digit
    min_E_in = min(E_in)
    min_E_in_digit = digits[E_in.index(min_E_in)]
    
    # find the maximum support vectors
    max_alpha_sum = max(alpha_sum_list)
    max_alpha_sum_digit = digits[alpha_sum_list.index(max_alpha_sum)]

    print(f"Min E_in: {min_E_in:.4f} for digit {min_E_in_digit}")
    print(f"Max support vectors: {max_alpha_sum} for digit {max_alpha_sum_digit}")
    print(alpha_sum_list)
         
    