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

# randomly choose 1000 data points from the training set as the validation set

def randomly_choose_validation_set(train_X, train_Y):
    # shuffle the data
    np.random.seed(0)
    idx = np.random.permutation(train_X.shape[0])
    train_X = train_X[idx]
    train_Y = train_Y[idx]
    # choose the first 1000 data points as the validation set
    valid_X = train_X[:1000]
    valid_Y = train_Y[:1000]
    train_X = train_X[1000:]
    train_Y = train_Y[1000:]
    return train_X, train_Y, valid_X, valid_Y


# SVM with normal kernel
# train 100 times find the best gamma

if __name__ == "__main__":
    train_file = 'features.train'
    test_file = 'features.test'
    train_X, train_Y = read_file(train_file)
    test_X, test_Y = read_file(test_file)

    C = 0.1
    gamma_list = [0.001, 0.01, 0.1, 1, 10]
    train_Y = convert_Y(train_Y, 0)  # 轉換標籤為 "0 vs not 0"

    # 初始化 gamma 出現次數的計數器
    gamma_count = [0] * len(gamma_list)

    for i in range(100):  # 執行 100 次隨機驗證
        # 隨機分割資料集為訓練集與驗證集
        train_X_split, train_Y_split, valid_X, valid_Y = randomly_choose_validation_set(train_X, train_Y)

        # 記錄此輪的 E_val
        E_valid = []
        for gamma in gamma_list:
            clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)  # 高斯核 SVM
            clf.fit(train_X_split, train_Y_split)  # 使用訓練子集進行訓練
            valid_pred = clf.predict(valid_X)  # 對驗證集進行預測
            E_valid.append(1 - accuracy_score(valid_Y, valid_pred))  # 計算驗證誤差 E_val

        # 找出當前回合的最佳 gamma（E_val 最小）
        min_E_valid = min(E_valid)
        best_gamma_index = E_valid.index(min_E_valid)  # 找出對應 gamma 的索引
        gamma_count[best_gamma_index] += 1  # 更新最佳 gamma 的計數

    # 找出出現最多次的 gamma
    most_selected_gamma = gamma_list[gamma_count.index(max(gamma_count))]
    print(f"Best gamma: {most_selected_gamma} appeared {max(gamma_count)} times")
