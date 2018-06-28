import autograd
import numpy
import scipy.linalg

iop = 0
if iop == 0:
   np = numpy
   npeinsum = numpy.einsum
   svd = scipy.linalg.svd
else:   
   np = autograd.numpy
   npeinsum = autograd.numpy.einsum
   svd = autograd.numpy.linalg.svd
