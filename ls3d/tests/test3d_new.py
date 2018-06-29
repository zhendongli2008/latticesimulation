#
# Test: 3D case in various directions (2018.06.28-)
#
# NOTE: 1. we can always choose a=1 as the unit of length!
#          then we observe how much fictious sites are required!
#       2. for small mass, a fairly large n is required,
#	   the estimate in latticeQED paper suggests n=1000.
#
# 	   This is hard to test for direct inversion, as the 
# 	   matrix size is L**3-by-L**3, but maybe can be pushed 
#	   to large n by 3D contraction.
#
import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from latticesimulation.ls2d import exact2d
from latticesimulation.ls3d import gen2d,genPEPO

m = 8 #10
n = 2*m+1
a = 1.0

# Test Z: for small m (longer-range), clearly large n is required!
for mass in [1.0,1.e-1,1.e-2,1.e-3]:
   t3d = exact2d.genT3d(n,mass*a)
   t3d = t3d.reshape((n**3,n**3))
   print t3d.shape
   tinv = scipy.linalg.inv(t3d)
   tinv = tinv.reshape((n,n,n,n,n,n))
   det = scipy.linalg.det(t3d)
   print '\nTest Z:'
   print 'direct detM=',det,' scale0=',pow(det,-1.0/n**3)
   
   fac = 4*numpy.pi
   def exactFun(x):
      return numpy.exp(-mass*x)/x/fac
   xmax = a*m*numpy.sqrt(3)
   x = numpy.linspace(0.01,xmax,100)
   y = exactFun(x)*fac
  
   plt.plot(x,1/x,'k--',label='1/r (reference)')
   plt.plot(x,y,'k-',label='exact (smooth) mass='+str(mass))
   tinv = tinv/a*fac
   xh = numpy.arange(m+1)*a
   plt.plot(xh,tinv[m,m,m][m,m,range(m,n)],'rx',
   	 label='z-direct: (m,m,n)-(m,m,m)')
   xd = numpy.arange(m+1)*a*numpy.sqrt(3)
   plt.plot(xd,tinv[m,m,m][range(m,n),range(m,n),range(m,n)],'b+',
   	 label='d-direct: (n,n,n)-(m,m,m)')
   plt.legend()
   plt.xlim(xmax=m)
   plt.ylim(ymin=-0.1,ymax=1.2)
   plt.show()

exit()


#=================
# TNS Contraction 
#=================
iprt = 1
auxbond = 100
print 'n=',n

# Only 2*2*2
import contraction3d
mass = 1.e-3
a = 1.0
n = 2
auxbond = 1000
t3d = exact2d.genT3d(n,mass*a)
t3d = t3d.reshape((n**3,n**3))
tinv = scipy.linalg.inv(t3d)
tinv = tinv.reshape((n,n,n,n,n,n))
result = initialization(n,mass**2,iprt,auxbond)
scale,zpeps0,zpeps,local2,local1a,local1b = result
print '\nTest Z:'
print 'direct detM=',scipy.linalg.det(t3d)
print 'scale=',scale,'Z=',numpy.power(1.0/scale,n*n*n)
print 'tr(T)=',contraction3d.contract(zpeps0,auxbond)

for i in range(n):
 for j in range(n):
  for k in range(n):
     epeps3d = zpeps.copy()
     s = epeps3d[i,j,k].shape
     epeps3d[i,j,k] = local2[:s[0],:s[1],:s[2],:s[3],:s[4],:s[5]]
     print '(i,j,k)=',(i,j,k),contraction3d.contract(epeps3d,auxbond),\
	       tinv[i,j,k,i,j,k]
