import numpy as np

def backward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3).T
    Y_assess = np.random.randn(3,)
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

    cache = {'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619],
         [-0.05225116,  0.02725659, -0.02646251],
         [-0.02009721,  0.0036869 ,  0.02883756],
         [ 0.02152675, -0.01385234,  0.02599885]]),
  'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]),
  'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ],
         [-0.05229879,  0.02726335, -0.02646869],
         [-0.02009991,  0.00368692,  0.02884556],
         [ 0.02153007, -0.01385322,  0.02600471]]),
  'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])}

    expected_output = {
    'dW1': np.array([[ 0.01018708, -0.00708701],
                [ 0.00873447, -0.0060768 ],
                [-0.00530847,  0.00369379],
                [-0.02206365,  0.01535126]]),
    'db1': np.array([[-0.00069728],[-0.00060606],[ 0.000364  ],[ 0.00151207]]),
    'dW2': np.array([[ 0.00363613,  0.03153604,  0.01162914, -0.01318316]]),
    'db2': np.array([[ 0.06589489]])}
    return parameters, cache, X_assess, Y_assess, expected_output
