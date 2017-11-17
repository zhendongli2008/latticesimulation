import numpy
import scipy.linalg
import matplotlib.pyplot as plt

n = 4 #10
m = n/2
t = numpy.zeros((n,n))
for i in range(n):
   t[i,i] = 2
for i in range(n-1):
   t[i,i+1] = -1
   t[i+1,i] = -1

print t

th = t[:m,:m]
e,vh = scipy.linalg.eigh(th)
v = numpy.zeros((n,n))
v[:m,:m] = vh
v[m:,m:] = vh

teff = v.T.dot(t.dot(v))
plt.matshow(teff)
plt.show()
print teff
