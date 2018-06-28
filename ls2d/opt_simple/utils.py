from include import np 
dtype=None

def zeros(shape):
    # allow us to set the default zero matrix type
    # (e.g. real or complex) by setting utils.dtype at beginning
    # of program
    return np.zeros(shape,dtype=dtype)
