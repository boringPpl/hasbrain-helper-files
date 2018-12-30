import numpy as np

def compute_cost_test_case():
    np.random.seed(1)
    Y_assess = np.random.randn(3, )
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'b2': np.array([[ 0.]])}

    a2 = (np.array([ 0.5002307 ,  0.49985831,  0.50023963]))

    return a2, Y_assess, parameters, 0.692919893776
