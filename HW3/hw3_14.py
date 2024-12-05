# create dataset for linear regression

import numpy as np
import random

def generate_sample_points():
    sample_points = []
    for i in range(1000):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        sample_points.append((x1,x2))
    return sample_points

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

# transform data into non-linear feature space
# (x0=1, x1, x2, x1x2, x1^2, x2^2)

def transform(sample_points):
    transformed_points = []
    for point in sample_points:
        x0 = 1
        x1 = point[0]
        x2 = point[1]
        x3 = x1*x2
        x4 = x1**2
        x5 = x2**2
        transformed_points.append((x0,x1,x2,x3,x4,x5))
    return transformed_points


# find the linear regression in the new feature space
# use Moore-Penrose to prevent singular matrix

def linear_regression(sample_points, labels):
    X = np.array([[point[0],point[1],point[2],point[3],point[4],point[5]] for point in sample_points])
    Y = np.array(labels)
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
    return w

# run the experiment for 1000 times and calculate the average Eout
# compare the result with hw3_13.py
# g1(x1,x2) = -1 - 0.05x1 + 0.08x2 + 0.13x1x2 + 15x1^2 + 1.5x2^2
# g2(x1,x2) = -1 - 1.5x1 + 0.08x2 + 0.13x1x2 + 1.5x1^2 + 15x2^2


if __name__ == '__main__':
    sample_points = generate_sample_points()
    labels = create_labels(sample_points)
    labels = flip_labels(labels)
    transformed_points = transform(sample_points)
    w = linear_regression(transformed_points,labels)
    print(w)