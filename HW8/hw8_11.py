import numpy as np
import numpy.random as rd
import sys
import matplotlib.pyplot as plt






def forward(x_list, w_list, s_list, layer, dim):
    """
    Forward pass for a fully connected neural network.
    
    Args:
        x_list: list of activations (including bias at index 0)
        w_list: list of weight matrices (including bias weights)
        s_list: list of pre-activations (s values, with dummy 0 at index 0)
        layer: total number of layers (input + hidden + output)
        dim: number of neurons per layer (no bias counted)
    
    Returns:
        None (modifies x_list and s_list in-place)
    """
    for l in range(1, layer):
        x_prev = x_list[l-1]
        # add bias
        if x_prev[0] != 1.0:
            x_prev = np.insert(x_prev, 0, 1.0)
        # calculate the activation
        W = w_list[l]

        s = np.dot(x_prev, W)

        x = np.tanh(s)

        # Add bias to activation for next layer
        x_with_bias = np.insert(x, 0, 1.0)

        # Add dummy 0 to s for consistency
        s_with_dummy = np.insert(s, 0, 0.0)

        # Save results
        x_list.append(x_with_bias)
        s_list.append(s_with_dummy)

