import numpy

n = 3
a = numpy.zeros([n,n],dtype=numpy.object)
for i in xrange(n):
  for j in xrange(n):
    a[i,j] = numpy.zeros([2,2])

b = numpy.zeros([n,n,2,2])
b[:] = numpy.vstack(a.flatten())
print b.shape

#print a.shape
#print numpy.einsum('ij',a).shape
#print numpy.asarray(a).shape
#print a.flatten().shape
#
#print numpy.vstack(a.flatten()).shape
#print numpy.hstack(a.flatten()).shape
