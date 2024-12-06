# import necessary function from hw3_14.py
from hw3_14 import *


# error rate function 

def error_rate(predictions, labels):
    error = 0
    for i in range(len(predictions)):
        if predictions[i] != labels[i]:
            error += 1
    return error / len(predictions)

# generating a new set of 1000 points and adding noise to the labels
# run the experiment for 1000 times and calculate the average Eout
# using g2 as the approximation for the target function

if __name__ == '__main__':
    Eout = 0  # Initialize the cumulative error
    for i in range(1000):
        # Generate sample points and corresponding labels
        sample_points = generate_sample_points()
        labels = create_labels(sample_points)
        labels = flip_labels(labels)
        
        # Calculate predictions using g2
        predictions = []
        for point in sample_points:
            predictions.append(g2(point[0], point[1]))
        
        # Accumulate the error rate
        Eout += error_rate(predictions, labels)

    # Calculate the average Eout
    print(Eout / 1000)
