import numpy as np
import numpy.random as rd
import sys
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



def forward(x_list, w_list, s_list, layer, dim):
    for i in range(1,layer):
        x_prev = x_list[i-1]
        # add bias
        if x_prev[0] != 1.0:
            x_prev = np.insert(x_prev, 0, 1.0)
            x_list[i-1] = x_prev
        # calculate the activation
        W = w_list[i]
        s = np.dot(x_prev, W)
        x = np.tanh(s)
        # Add bias to activation for next layer

        x_with_bias = np.insert(x, 0, 1.0)
        # Add dummy 0 to s for consistency
        s_with_dummy = np.insert(s, 0, 0.0)
        # Save results
        x_list.append(x_with_bias)
        s_list.append(s_with_dummy)
        


def tanh_derivative(s):
    return 1 - np.tanh(s)**2



def backward(delta_list, w_list, s_list, x_list, y, layer, dim):

    # initialize delta list
    for l in range(1, layer):
        # add dummy 0 at index 0
        delta = np.zeros(dim[l] + 1) 
        delta_list.append(delta)

    # Output layer delta : dE/ds = -2(y - ŷ) * tanh'(s)
    # bias node always 1 , so no need to calculate their delta
    s_last = s_list[layer-1][1:]     
    x_last = x_list[layer-1][1:]      
    delta_last = -2 * (y - x_last) *  tanh_derivative(s_last)  
    delta_list[layer-1][1:] = delta_last              

    # Backpropagate
    for l in reversed(range(1, layer - 1)):  # from layer-2 to 1
        # Get weight matrix for layer l+1 and remove bias row
        # earlier delta won't affect bias of latter layer
        W_next = w_list[l+1][1:, :]          # shape: (dim[l], dim[l+1])
        delta_next = delta_list[l+1][1:]     # next layer delta (shape: dim[l+1])
        s_curr = s_list[l][1:]               # pre-activation of current layer

        # vectorized delta calculation
        delta_curr = np.dot(W_next, delta_next) * tanh_derivative(s_curr)  # shape: (dim[l],)
        delta_list[l][1:] = delta_curr  # assign back to delta list

def cal_gd(gd_list, x_list, delta_list, layer):
    for l in range(1, layer):
        # Gradient = (input from previous layer)^T × (error from current layer)
        gd = np.array(np.dot(np.matrix(x_list[l-1]).transpose(), np.matrix(delta_list[l][1:])))
        # Save the calculated gradient
        gd_list.append(gd)


def nnet_bp(data_x, data_y, layer, dim, eta, r, T=5000):
    """
    Train a simple fully connected neural network using backpropagation and SGD.
    """
    # Convert dim list into numpy array for easy indexing
    dim = np.array([int(i) for i in dim])

    # Initialize weight list; w_list[0] is empty (no weights for input layer)
    w_list = [np.array([])]

    # Randomly initialize weights for each layer (include bias weight)
    for l in range(1, layer):
        w_list.append(rd.uniform(-1*r, r, size=(dim[l-1]+1, dim[l])))

    # Main training loop: run T iterations
    for t in range(T):
        # Prepare input: randomly pick one training example (SGD)
        x_list = []
        n = rd.randint(0, data_x.shape[0])
        x_list.append(data_x[n])

        # Initialize lists to store intermediate values
        s_list, delta_list, gd_list = [np.array([])], [np.array([])], [np.array([])]

        # 1. Forward pass: calculate all layer outputs and pre-activations
        forward(x_list, w_list, s_list, layer, dim)

        # 2. Backward pass: calculate delta (error signal) for each layer
        backward(delta_list, w_list, s_list, x_list, data_y[n], layer, dim)

        # 3. Calculate gradient for each layer based on x and delta
        cal_gd(gd_list, x_list, delta_list, layer)

        # 4. Update weights using calculated gradients
        for l in range(1, layer):
            w_list[l] = w_list[l] - eta * gd_list[l]

    # Return the trained weight list
    return w_list


def nnet_predict(data_x, w_list, layer, dim):
    #  Prepare an empty array to store prediction results
    y_pred = np.zeros(data_x.shape[0])  
    
   
    for i in range(data_x.shape[0]):
        #Prepare input for this sample
        x_list = list([])             
        x_list.append(data_x[i])     
        # Prepare empty list to store pre-activations (s values)
        s_list = [np.array([])]        # s_list[0] dummy empty array
        
        # Forward pass - calculate outputs of all layers
        forward(x_list=x_list, w_list=w_list, s_list=s_list, layer=layer, dim=dim)
        
        # Store the output prediction
        y_pred[i] = x_list[-1][1]      # Output layer's real prediction (skip bias at index 0)
    
    return y_pred


def get_sign(d):
    return np.where(d > 0, 1, -1)

def get_err(sign_prim, sign_pred):
    if sign_prim.shape[0] != sign_pred.shape[0]:
        raise ValueError("Shape mismatch between labels and predictions")
    
    # Count number of mismatches
    err_count = np.sum(sign_prim != sign_pred)
    return err_count / sign_prim.shape[0]


if __name__ == "__main__":
    # 1. 讀入訓練與測試資料
    file_train = 'nnet_train.dat'
    file_test = 'nnet_test.dat'
    train_x, train_y = read_file(file_train)
    test_x, test_y = read_file(file_test)

    # 2. 固定參數
    r = 0.1           # 初始化範圍
    eta = 0.01        # 固定學習率
    T = 50000         # 每次訓練迭代次數
    M1 = 8           # 隱藏層神經元數
    M2 = 3           # 隱藏層神經元數
    trials = 500      # 重複實驗次數
    layer_n = 4       # 四層網路（含輸入與輸出）
    
    # 根據 d-8-3-1 結構設定維度（dim[0] 是輸入層神經元數）
    dim = np.zeros(layer_n)
    dim[0] = train_x.shape[1]  # 輸入維度
    dim[1] = M1
    dim[2] = M2
    dim[3]= 1                 # 輸出層為1個
    dim = dim.astype(int)
    # 儲存每次實驗的測試錯誤
    err_trials = np.zeros(trials)

    for t in range(trials):
        # 訓練神經網路
        w_list = nnet_bp(train_x, train_y, layer=layer_n, dim=dim, eta=eta, r=r, T=T)

        # 在測試資料上預測
        y_pred = nnet_predict(data_x=test_x, w_list=w_list, layer=layer_n, dim=dim)

        # 計算錯誤率（例如 0/1 錯誤）
        err = get_err(test_y, get_sign(y_pred))
        err_trials[t] = err

    # 平均測試錯誤率
    err_avg = np.mean(err_trials)
    print(f"Average E_out over {trials} trials = {err_avg}")
