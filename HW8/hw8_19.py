import numpy as np
import sys
import numpy.random as rd
import matplotlib.pyplot as plt


def read_file(file_name):
    with open(file_name, 'r') as file:
        data = file.readlines()
    x = []
    for line in data:
        line = line.strip().split()
        x.append([float(i) for i in line]) 
    return np.array(x)


# implement k-means algorithm
def kmeans(data, k, max_iter=100, tol=1e-4):
    N, d = data.shape
    # Step 1: 初始化中心
    idx = np.random.choice(N, k, replace=False)
    centroids = data[idx]

    labels = np.zeros(N, dtype=int)

    for _ in range(max_iter):
        # Step 2: 分配群集（assignment step）
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)**2  # shape: (N, k)
        new_labels = np.argmin(distances, axis=1)

        # Step 3: 收斂判斷
        if np.all(new_labels == labels):
            break
        labels = new_labels

        # Step 4: 更新中心（update step）
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                # 若某一群為空 → 隨機選一點重新初始化
                centroids[i] = data[np.random.choice(N)]

    return centroids, labels


def get_error(data, labels, centroids):
    N = data.shape[0]
    total = 0
    for i in range(N):
        center = centroids[labels[i]]          # 找到該資料點所屬的中心
        diff = data[i] - center                 # 計算差距向量
        dist_sq = np.sum(diff ** 2)             # 計算距離平方
        total += dist_sq
    return total / N                            # 平均距離平方（E_in）


if __name__ == '__main__':
    # 讀取資料（只需要 x，不用 y）
    data = read_file('kmeans_train.dat')

    k_list = [2, 10]
    num_trials = 500

    for k in k_list:
        error_list = []

        for _ in range(num_trials):
            centroids, labels = kmeans(data, k)
            error = get_error(data, labels, centroids)
            error_list.append(error)

        avg_error = np.mean(error_list)
        print(f'k = {k}, Average E_in over {num_trials} runs: {avg_error:.4f}')
