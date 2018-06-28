import numpy
import scipy.linalg

n = 20
kij = numpy.zeros((n,n))
for i in range(n):
   kij[i,i] = 2.0
for i in range(n-1):
   kij[i,i+1] = -1.0
   kij[i+1,i] = -1.0

kbd = kij[:8,:8]
#print scipy.linalg.inv(kbd)
#exit()

e,u = scipy.linalg.eigh(kbd)
u = scipy.linalg.block_diag(u,numpy.identity(4),u)
print u.shape
print u.T.dot(kij.dot(u))[8:14,8:14]
