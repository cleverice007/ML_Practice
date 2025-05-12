import numpy as np
import sys
import numpy.random as rd
import matplotlib.pyplot as plt

# read file and split data into x and y
def read_file(file_name):
    with open(file_name, 'r') as file:
        data = file.readlines()
    x = []
    Y = []
    for line in data:
        line = line.strip().split()
        x.append([float(i) for i in line[:-1]])
        Y.append(int(line[-1]))
    return np.array(x), np.array(Y)

# implement knn algorithm

def knn(x_train, y_train, x_test, k):

    y_pred = []

    for i in range(x_test.shape[0]):
        # Compute vector of squared distances to all training points
        diff = x_train - x_test[i]                   # shape: (n_train, d)
        dist = np.sum(diff**2, axis=1)               # shape: (n_train,), squared Euclidean

        # Get indices of k nearest neighbors
        knn_idx = np.argsort(dist)[:k]               # sorted by distance
        knn_labels = y_train[knn_idx]                # get their labels

        # Count occurrences of each label
        unique, counts = np.unique(knn_labels, return_counts=True)
        label_count = dict(zip(unique, counts))
        max_count = max(label_count.values())

        # Find all labels tied for max vote
        top_labels = [label for label, count in label_count.items() if count == max_count]

        if len(top_labels) == 1:
            # If no tie, assign the winning label
            y_pred.append(top_labels[0])
        else:
            # If tie, choose the label of the closest neighbor among the top labels
            for idx in knn_idx:
                if y_train[idx] in top_labels:
                    y_pred.append(y_train[idx])
                    break

    return np.array(y_pred)

def get_err(sign_prim, sign_pred):
    if sign_prim.shape[0] != sign_pred.shape[0]:
        raise ValueError("Shape mismatch between labels and predictions")
    
    # Count number of mismatches
    err_count = np.sum(sign_prim != sign_pred)
    return err_count / sign_prim.shape[0]

if __name__ == '__main__':
    # 1. 讀取資料
    train_x, train_y = read_file('knn_train.dat')
    test_x, test_y = read_file('knn_test.dat')

    # 2. 設定要測試的 k 值
    k_list = [1, 5]

    # 3. 初始化錯誤率儲存陣列
    E_in = []
    E_out = []

    # 4. 對每個 k 值分別計算錯誤率
    for k in k_list:
        # 訓練預測與錯誤率
        y_train_pred = knn(train_x, train_y, train_x, k)
        E_in_k = get_err(train_y, y_train_pred)

        # 測試預測與錯誤率
        y_test_pred = knn(train_x, train_y, test_x, k)
        E_out_k = get_err(test_y, y_test_pred)

        # 儲存結果
        E_in.append(E_in_k)
        E_out.append(E_out_k)

        # 輸出每個 k 的結果
        print(f"k = {k}:")
        print(f"  E_in  = {E_in_k:.3f}")
        print(f"  E_out = {E_out_k:.3f}")

