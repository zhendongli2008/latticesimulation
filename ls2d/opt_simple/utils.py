import autograd.numpy as N
dtype=None

def zeros(shape):
    # allow us to set the default zero matrix type
    # (e.g. real or complex) by setting utils.dtype at beginning
    # of program
    return N.zeros(shape,dtype=dtype)
