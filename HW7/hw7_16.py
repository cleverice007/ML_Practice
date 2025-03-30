import numpy as np


# read file and split data into x and y
def read_file(file_name):
    with open(file_name, 'r') as file:
        data = file.readlines()
    x = []
    y = []
    for line in data:
        line = line.strip().split()
        x.append([float(i) for i in line[:-1]])
        y.append(int(line[-1]))
    return np.array(x), np.array(y)




# input argument: array of integers
# if value <= 0 return -1, if value > 0 return 1
def get_sign(values):
    return np.where(values <= 0, -1, 1)


def get_gini_index(s, theta, feature, X, Y):
    left_Y = []
    right_Y = []
    for i, x in enumerate(X):  
        if s * (x[feature] - theta) < 0:
            left_Y.append(Y[i])
        else:
            right_Y.append(Y[i])

    # if left_Y or right_Y is empty, gini index is 0
    if len(left_Y) == 0:
        left_gini = 0
    else:
        left_Y = np.array(left_Y)
        p_pos = np.sum(left_Y == 1) / len(left_Y)
        p_neg = np.sum(left_Y == -1) / len(left_Y)
        left_gini = 1 - p_pos**2 - p_neg**2

    if len(right_Y) == 0:
        right_gini = 0
    else:
        right_Y = np.array(right_Y)
        p_pos = np.sum(right_Y == 1) / len(right_Y)
        p_neg = np.sum(right_Y == -1) / len(right_Y)
        right_gini = 1 - p_pos**2 - p_neg**2

    return (len(left_Y) / len(Y)) * left_gini + (len(right_Y) / len(Y)) * right_gini


# iterate all features, s, theta to calculate gini index
# return the best split

def get_theta(x):
    x = np.sort(np.unique(x))
    eps = 1e-5  
    mid_theta = (x[1:] + x[:-1]) / 2
    left_bound = x[0] - eps
    right_bound = x[-1] + eps
    return np.concatenate([[left_bound], mid_theta, [right_bound]])


def branch_criterion(X, Y):
    s_list = [-1, 1]
    min_gini = np.inf
    best_s, best_theta, best_feature = None, None, None
    for s in s_list:
        for feature in range(X.shape[1]):
            theta_list = get_theta(X[:, feature])
            for theta in theta_list:
                gini = get_gini_index(s, theta, feature, X, Y)
                if gini < min_gini:
                    min_gini = gini
                    best_s = s
                    best_theta = theta
                    best_feature = feature
    return best_s, best_theta, best_feature, min_gini

class Node(object):
    # store the value of the node and the left and right child
    def __init__(self, val=(0, 0, 0)):
       self.val = val
       self.left = None
       self.right = None
    
    def is_leaf(self):
        return self.val[0] == 0

    def set_left(self, left):
        self.left = left
    def set_right(self, right):
        self.right = right

    def __repr__(self):
        if self.is_leaf():
            return f"Leaf Node: Predict = {self.val[1]}"
        else:
            return f"Internal Node: s = {self.val[0]}, theta = {self.val[1]}, feature = {self.val[2]}"


class CaRTree(object):
    def __init__(self):
        self.root = None

    def build_cart(self, X, Y):
        self.root = self.generate_cart(X, Y)

    def generate_cart(self, X, Y):
        if len(set(Y)) == 1:
            return Node((0, Y[0], 0))
        else:
            s, theta, feature, gini = branch_criterion(X, Y)
            node = Node((s, theta, feature))
            # split the data to left and right
            res = s * get_sign(X[:,feature] - theta)
            left_idx = (res == -1)
            right_idx = (res == 1)
            node.set_left(self.generate_cart(X[left_idx], Y[left_idx]))
            node.set_right(self.generate_cart(X[right_idx], Y[right_idx]))
        return node


    def get_val(self, x):
        node = self.root
        while not node.is_leaf():
            s, theta, feature = node.val
            res = s * get_sign(x[feature] - theta)
            if res == -1:
                node = node.left
            else:
                node = node.right
        return node.val[1]


    def predict(self, X):
        return [self.get_val(x) for x in X]



if __name__ == "__main__":
    # read train and test data
    train_X, train_Y = read_file('train.dat')
    test_X, test_Y = read_file('test.dat')

    # build tree
    cart = CaRTree()
    cart.build_cart(train_X, train_Y)

    # predict ein and eout
    ein = np.mean(cart.predict(train_X) != train_Y)
    eout = np.mean(cart.predict(test_X) != test_Y)


    # print the result
    print(f"Ein: {ein}")
    print(f"Eout: {eout}")

