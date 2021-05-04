import numpy as np

def normalize(array: np.array) -> np.array:
    '''
    Input:
        array: a numpy array of dimension m x n, where m := number of samples and n := vector size.
    '''
    return (array - array.mean(axis = 1).reshape(len(array), 1))/(array.std(axis = 1).reshape(len(array), 1))