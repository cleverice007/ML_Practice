# import necessary function from hw3_14.py
from hw3_14 import *


#  output of the g2 is sign of g2(x1,x2) = -1 - 1.5x1 + 0.08x2 + 0.13x1x2 + 0.05x1^2 + 1.5x2^2
#  calculate error rate by comparing the prediction of g2 and the labels

def error_rate(sample_points, labels):
    error = 0
    for i in range(len(sample_points)):
        if g2(sample_points[i][0], sample_points[i][1]) != labels[i]:
            error += 1
    return error / len(sample_points)

# generating a new set of 1000 points and adding noise to the labels
# run the experiment for 1000 times and calculate the average Eout
# using g2 as the approximation for the target function

if __name__ == '__main__':
    Eout = []  # Initialize the cumulative error
    for i in range(1000):
        # Generate sample points and corresponding labels
        sample_points = generate_sample_points()
        labels = create_labels(sample_points)
        labels = flip_labels(labels)
        
        # Calculate predictions using g2
        Eout.append(error_rate(sample_points, labels))

    # Calculate the average Eout
    print(sum(Eout) / 1000)