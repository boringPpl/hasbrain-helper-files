import numpy as np
import h5py

def load_dataset():
    '''
    Load dataset from h5 file to numpy array
    Returns:
    - A tuple: (training data array, training label array, test data array, test label array, number of class)
    '''
    train_dataset = h5py.File('input/catnnoncat/cat_n_noncat/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('input/catnnoncat/cat_n_noncat/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_standardlized_dataset():
    '''
    Load the dataset and preprocess it
    Returns:
    - A tuple: (standardlized training data array, training label array, standardlized test data array, test label array, number of class)
    '''
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    test_set_x_flatten =  test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.
    return test_set_x, train_set_y, test_set_x, test_set_y, classes

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    return 1/(1+np.exp(-z))
