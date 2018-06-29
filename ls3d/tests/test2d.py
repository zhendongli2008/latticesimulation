#
# Test: 2D case in various directions 
#
import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from latticesimulation.ls2d import exact2d
from latticesimulation.ls3d import gen2d,genPEPO

m = 10
n = 2*m+1
a = 1.0
# note that error is large for small mass, 
# which requires a very large lattice size!
mass = 0.1
t2d = exact2d.genT2d(n,mass)
t2d = t2d.reshape((n*n,n*n))
print t2d.shape
tinv = scipy.linalg.inv(t2d)
tinv = tinv.reshape((n,n,n,n))
#plt.matshow(tinv[m,m])
#plt.show()

def exactFun(x):
   return scipy.special.k0(mass*x)/(2*numpy.pi)
x = numpy.linspace(0.01,m*numpy.sqrt(2),100)
y = exactFun(x)

#gamma=0.5772156649015329
#plt.plot(x,(-numpy.log(mass*x/2)-gamma)/2.0/numpy.pi,'ro-',label='asym')
plt.plot(x,y,'k-',label='exact (smooth): n='+str(n))
plt.plot(tinv[m,m][m,range(m,n)],'gx',label='h-direct: (m,n)-(m,n)')
plt.plot(tinv[m,m][range(m,n),m],'b+',label='v-direct: (n,m)-(m,m)')
xd = numpy.arange(m+1)*numpy.sqrt(2)
plt.plot(xd,tinv[m,m][range(m,n),range(m,n)],'ro',label='d-direct: (n,n)-(m,m)')
plt.legend()
plt.ylim(ymin=-0.1)
plt.show()

#
# Numerical test
#
iprt = 1
abond = 40
mass2 = mass**2
scale,zpeps,local2,local1a,local1b = gen2d.initialization(n,mass2,iprt,abond)
det = scipy.linalg.det(t2d)
print 
print ' lattice size=',(n,n),' matrix size=',(n**2,n**2)
print ' det(t2d)=',det,' pow(det,-1/lsize)=',pow(det,-1.0/n**2),' scale=',scale
scale0 = pow(det,-1.0/n**2)
scale,zpeps,local2,local1a,local1b = gen2d.initialization(n,mass2,iprt,abond,guess=scale0)
exit()

npepo = genPEPO.genNPEPO(n,mass2,iprt=0,auxbond=abond)
numpy.random.seed(0)
print '\nRandomCheck:'
for i in range(10):
   rint = numpy.random.randint(1,high=n-1,size=4)
   palst = [(rint[0],rint[1])]
   pblst = [(rint[2],rint[3])]
   v0 = tinv[palst[0]][pblst[0]]
   for abond in [10,20,30,40]:
      v1 = genPEPO.pepo2cpeps(npepo,palst,pblst,auxbond=abond)[0,0]
      print i,palst,pblst,'v0=',v0,'Dcut=',abond,'v1=',v1,'diff=',v0-v1
   print

plt.legend()
plt.show()
