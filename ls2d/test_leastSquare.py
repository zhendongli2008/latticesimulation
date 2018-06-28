import numpy
import scipy.linalg

# Weighted
amat = numpy.load('amat.npy')
bvec = numpy.load('bvec.npy')
  
print amat.shape
print numpy.max(abs(amat)),numpy.min(abs(amat))
print numpy.max(abs(bvec)),numpy.min(abs(bvec))

atmp = amat.T.dot(amat)
btmp = amat.T.dot(bvec)

e,v = numpy.linalg.eigh(atmp)
print 'e=',e

# (A^T*A)*c=A^T*b
clst = numpy.linalg.solve(atmp,btmp)
print "CLST0=",clst
errs = numpy.linalg.norm(atmp.dot(clst)-btmp)
print 'err0=',errs
errs = numpy.linalg.norm(amat.dot(clst)-bvec)
print 'err0=',errs

# A+*b
clst1 = scipy.linalg.pinv(amat).dot(bvec)
print "CLST1=",clst1
print numpy.linalg.norm(clst-clst1)
errs = numpy.linalg.norm(atmp.dot(clst1)-btmp)
print 'err0=',errs
errs = numpy.linalg.norm(amat.dot(clst)-bvec)
print 'err0=',errs

# lstsq
clst1 = scipy.linalg.lstsq(amat,bvec)[0]
print "CLST1=",clst1
print numpy.linalg.norm(clst-clst1)
errs = numpy.linalg.norm(atmp.dot(clst1)-btmp)
print 'err0=',errs
errs = numpy.linalg.norm(amat.dot(clst)-bvec)
print 'err0=',errs

import scipy.optimize
nparams = amat.shape[1]
bnds = (numpy.zeros(nparams),numpy.array([numpy.inf]*nparams))
clst1 = scipy.optimize.lsq_linear(amat,bvec,bounds=bnds).x
print "CLST1=",clst1
print numpy.linalg.norm(clst-clst1)
errs = numpy.linalg.norm(atmp.dot(clst1)-btmp)
print 'err0=',errs
errs = numpy.linalg.norm(amat.dot(clst)-bvec)
print 'err0=',errs
