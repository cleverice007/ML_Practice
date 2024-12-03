# generate 1,000sample points (x1,x2)
#  ～ uniform distribution in [-1,1] x [-1,1]
import numpy as np
import random

def generate_sample_points():
    sample_points = []
    for i in range(1000):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        sample_points.append((x1,x2))
    return sample_points

# target function : f(x1,x2) = sign(x1^2 + x2^2 - 0.6)
# create labels for the sample points
# flip the sign of the output with 10% probability
def target_function(x1,x2):
    return np.sign(x1**2 + x2**2 - 0.6)

def create_labels(sample_points):
    labels = []
    for point in sample_points:
        labels.append(target_function(point[0],point[1]))
    return labels
def flip_labels(labels):
    for i in range(len(labels)):
        if random.random() < 0.1:
            labels[i] = -labels[i]
    return labels

# carry out linear regression with (x0=1, x1, x2) to find g
# calculate error rate 
def linear_regression(sample_points, labels):
    X = np.array([[1,point[0],point[1]] for point in sample_points])
    Y = np.array(labels)
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
    return w
def error_rate(w,sample_points,labels):
    error = 0
    for i in range(len(sample_points)):
        if np.sign(np.dot(w,[1,sample_points[i][0],sample_points[i][1]])) != labels[i]:
            error += 1
    return error/len(sample_points)

# run the experiment for 1000 times and calculate the average Eout

if __name__ == '__main__':
    Eout = 0
    for i in range(1000):
        sample_points = generate_sample_points()
        labels = create_labels(sample_points)
        labels = flip_labels(labels)
        w = linear_regression(sample_points,labels)
        Eout += error_rate(w,sample_points,labels)
    print(Eout/1000)