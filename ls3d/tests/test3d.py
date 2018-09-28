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
from latticesimulation.ls2d import exact2d
from latticesimulation.ls3d import gen2d,gen3d,genPEPO

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2 
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
fsize = 20
msize = 14

# 5-by-5 is a resonable test

m = 7 #3 #4-inf #8 #10
n = 2*m+1
print 'lattice n=',n
print 't3d.shape=',(n**3,n**3)
print

# Test Z: for small m (longer-range), clearly large n is required!
for idx,mass in enumerate([1.0,1.e-1,1.e-2,1.e-3]):
   t3d = exact2d.genT3d(n,mass)
   t3d = t3d.reshape((n**3,n**3))
   tinv = scipy.linalg.inv(t3d)
   tinv = tinv.reshape((n,n,n,n,n,n))
   det = scipy.linalg.det(t3d)
   print '\nTest Z with mass =',mass
   print 'direct detM=',det,' scale0=',pow(det,-1.0/n**3)
   
   fac = 4*numpy.pi
   def exactFun(x):
      return numpy.exp(-mass*x)/x/fac
   xmax = (m+1)*numpy.sqrt(3)
   x = numpy.linspace(0.01,xmax,100)
   y = exactFun(x)*fac
   
   plt.subplot(221+idx)
   plt.plot(x,1/x,'k--',label='1/r (reference)')
   plt.plot(x,y,'k-',label='Yukawa mass='+str(mass))
   
   tinv = tinv*fac
   xz = numpy.arange(1,m+1)
   yz = tinv[m,m,m][m,m,range(m+1,n)]
   plt.plot(xz,yz,'rx',
	 markeredgewidth=1,\
	 markersize=10,linewidth=1,\
   	 label='z-direct: (m,m,n)-(m,m,m)')

   xd = numpy.arange(1,m+1)*numpy.sqrt(3)
   yd = tinv[m,m,m][range(m+1,n),range(m+1,n),range(m+1,n)]
   plt.plot(xd,yd,'b+',
	 markeredgewidth=1,\
	 markersize=10,linewidth=1,\
   	 label='d-direct: (n,n,n)-(m,m,m)')

   plt.xlim(xmax=numpy.max(xd+1.0))
   plt.ylim(ymin=-0.1,ymax=1.2)
  
#   #=================
#   # TNS Contraction 
#   #=================
#   iprt = 1
#   auxbond = 50
#   iop = 0
#   if iop == 0:
#      result = gen3d.initialization(n,mass**2,iprt,auxbond)
#      scale,zpeps,local2,local1a,local1b = result
#      gen3d.tensor_dump(scale,zpeps,local2,local1a,local1b)
#   else:
#      scale,zpeps,local2,local1a,local1b = gen3d.tensor_load()
#      zval = gen3d.test_zdir(m,n,scale,zpeps,local2,local1a,local1b,auxbond,off=1)
#      print
#      print 'zval0=',yz[1:]
#      print 'zval1=',zval,zval*fac
#      dval = gen3d.test_ddir(m,n,scale,zpeps,local2,local1a,local1b,auxbond,off=1)
#      print
#      print 'dval0=',yd[1:]
#      print 'dval1=',dval,dval*fac
#   exit()

# final   
#plt.legend()
plt.show()
