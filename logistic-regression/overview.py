import numpy as np
def test_sigmoid():
    test_inputs = [
        np.array([[1,2],[3,4]]),
        np.array([[-1,3],[13,4],[0,1]]),
        np.array([3,4,5]),
        np.array([[1,2,3,4],[3,4,5,6]]),
        np.array([3,4]),
        np.array([[1,2],[13,41],[23,45],[113,24]]),
        np.array([3,4,0,0]),
        np.array([[1,2,2],[3,4,2]]),
        np.array([13,24])
    ]

    expected_outputs = [
        np.array([[ 0.73105858,  0.88079708],
           [ 0.95257413,  0.98201379]]),
        np.array([[ 0.26894142,  0.95257413],
           [ 0.99999774,  0.98201379],
           [ 0.5       ,  0.73105858]]),
        np.array([ 0.95257413,  0.98201379,  0.99330715]),
        np.array([[ 0.73105858,  0.88079708,  0.95257413,  0.98201379],
           [ 0.95257413,  0.98201379,  0.99330715,  0.99752738]]),
        np.array([ 0.95257413,  0.98201379]),
        np.array([[ 0.73105858,  0.88079708],
           [ 0.99999774,  1.        ],
           [ 1.        ,  1.        ],
           [ 1.        ,  1.        ]]),
        np.array([ 0.95257413,  0.98201379,  0.5       ,  0.5       ]),
        np.array([[ 0.73105858,  0.88079708,  0.88079708],
           [ 0.95257413,  0.98201379,  0.88079708]]),
        np.array([ 0.99999774,  1.        ])
    ]
    return test_inputs, expected_outputs
