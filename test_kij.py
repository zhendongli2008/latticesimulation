import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from zmpo_dmrg.source.mpsmpo.mpo_class import class_mpo
from zmpo_dmrg.source.mpsmpo.mps_class import class_mps

Aref = 1.0 #1.071925
kappa = 0.5 #1.0/2.385345 # eq(7) of prb91,235151(2015)

def testVmpo(n,a):
   sfac = 1.0/(1.0+0.5*kappa0**2*a**2)
   tij = numpy.diag([1.0]*n)
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -0.5*sfac
   # V = 1/2*ni*Vij*nj
   def genVij(i,j):
      occ = numpy.zeros(2*n)
      if i == j:
         occ[2*(i-1)] = 1.0
         occ[2*(i-1)+1] = 1.0
      else:
         occ[2*(i-1)] = 1.0
         occ[2*(j-1)] = 1.0
      mps = class_mps(2*n)
      mps.hfstate(2*n,occ)
      mps = mps.merge([[2*i,2*i+1] for i in range(n)])
      val = mps.dot(vmpo.dotMPS(mps))
      return val
   def genVijDiag(i):
      occ = numpy.zeros(2*n)
      occ[2*(i-1)] = 1.0
      occ[2*(i-1)+1] = 1.0
      mps = class_mps(2*n)
      mps.hfstate(2*n,occ)
      mps = mps.merge([[2*i,2*i+1] for i in range(n)])
      val = mps.dot(vmpo.dotMPS(mps))
      return val
   # Plot
   assert n%2 == 1
   nc = (n-1)/2
   x = numpy.arange(-nc,nc+1)*a
   # Exact OBC: [0]_1_2_3_|_5_6_7_[8] nsite=7
   L = (n+1)*a
   z = Aref*numpy.sinh(kappa0*(x+L/2))*\
       numpy.sinh(kappa0*(L/2-x))/\
       (kappa0*numpy.sinh(kappa0*L))
   plt.plot(x,z,'b--',label='Exact_OBC')
   # Numerical
   for ng,style in zip([60,40,20,10],['k-','r-','b-','g-']):
      vmpo = genEVmpo(n,a,ng)
      vij_mid = numpy.array([genVij(int(n/2),i) for i in range(n)])
      plt.plot(x,vij_mid,style,label='Num_OBC (ng='+str(ng)+')')
   # New
   for ng,style in zip([60,40,20],['k+-','r+-','b+-']):
      vmpo = genEVmpoTaylor(n,a,ng)
      vij_mid = numpy.array([genVij(int(n/2),i) for i in range(n)])
      plt.plot(x,vij_mid,style,label='Num_OBC (ng='+str(ng)+')')
   plt.legend()
   plt.show()
   return 0

if __name__ == '__main__':
   test()
