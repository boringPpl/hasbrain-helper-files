import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

np.random.seed(1) # set a seed so that the results are consistent

def layer_sizes_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(3, 5)
    Y_assess = np.random.randn(3, 2)
    return X_assess, Y_assess

def initialize_parameters_test_case():
    n_x, n_h, n_y = 2, 4, 1
    W1 = np.array([[-0.00416758, -0.00056267], [-0.02136196,  0.01640271], [-0.01793436, -0.00841747], [ 0.00502881, -0.01245288]])
    b1 = np.array([[ 0.], [ 0.], [ 0.], [ 0.]])
    W2 = [[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]
    b2 = [[0.]]
    return n_x, n_h, n_y, W1, b1, W2, b2
