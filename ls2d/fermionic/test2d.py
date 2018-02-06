import numpy
import scipy.linalg
from latticesimulation.ls2d import exact2d
import matplotlib.pyplot as plt

m = 5
n = 2*m+1
a = 1.0
# note that error is large for small mass, 
# which requires a very large lattice size!
mass = 1.0 #0.04
t2d = exact2d.genT2d(n,mass)
t2d = t2d.reshape((n*n,n*n))
tinv = scipy.linalg.inv(t2d)
tinv = tinv.reshape((n,n,n,n))

#plt.matshow(tinv[m,m])
#plt.show()

def exactFun(x):
   return scipy.special.k0(mass*x)/(2*numpy.pi)

print t2d.shape
      
x = numpy.linspace(0.01,m*numpy.sqrt(2),1000)
y = exactFun(x)

plt.plot(x,y,'k-',label='exact')
plt.plot(tinv[m,m][m,range(m,n)],'go',label='horizontal')
plt.plot(tinv[m,m][range(m,n),m],'bo',label='vertical')
xd = numpy.arange(m+1)*numpy.sqrt(2)
plt.plot(xd,tinv[m,m][range(m,n),range(m,n)],'ro',label='diagonal')
plt.legend()
plt.show()

print t2d
print 'detM=',scipy.linalg.det(t2d)
exit()
