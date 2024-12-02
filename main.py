import torch
from helper import load_data
from solution import AE, frobeniu_norm_error
import numpy as np
import os



def test_ae(A, p):
    model = AE(d_hidden_rep=p)
    model.train(A, A, 128, 300)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w

if __name__ == '__main__':
    dataloc = "./data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    ### YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas
    ps = [10, 50, 100, 200]
    for p in ps:
        final_w = test_ae(A, p)  
    ### END YOUR CODE 
