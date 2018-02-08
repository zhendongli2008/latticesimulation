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
x = numpy.linspace(0.01,m*numpy.sqrt(2),1000)
y = exactFun(x)

plt.plot(x,y,'k-',label='exact')
plt.plot(tinv[m,m][m,range(m,n)],'go',label='horizontal')
plt.plot(tinv[m,m][range(m,n),m],'bo',label='vertical')
xd = numpy.arange(m+1)*numpy.sqrt(2)
plt.plot(xd,tinv[m,m][range(m,n),range(m,n)],'ro',label='diagonal')

#
# Numerical test
#
import genPEPO
abond = 30
npepo = genPEPO.genNPEPO(n,mass**2,iprt=0,auxbond=abond)
numpy.random.seed(0)
print '\nRandomCheck:'
for i in range(10):
   rint = numpy.random.randint(1,high=n-1,size=4)
   palst = [(rint[0],rint[1])]
   pblst = [(rint[2],rint[3])]
   v0 = tinv[palst[0]][pblst[0]]
   for abond in [10,20,30]:
      v1 = genPEPO.pepo2cpeps(npepo,palst,pblst,auxbond=abond)[0,0]
      print i,palst,pblst,'v0=',v0,'Dcut=',abond,'v1=',v1,'diff=',v0-v1
   print
exit()

plt.legend()
plt.show()
