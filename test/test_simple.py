import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from zmpo_dmrg.source.mpsmpo.mpo_class import class_mpo
from zmpo_dmrg.source.mpsmpo.mps_class import class_mps


def test():
   kappa0 = 5
   L = 20
   nlst = []
   vlst = []
   for n in range(10,1000,50):
      a = L/float(n+1)
      # Quadrature
      sfac = 1.0/(1.0+0.5*kappa0**2*a**2)
      tij = numpy.diag([1.0]*n)
      for i in range(n-1):
         tij[i,i+1] = tij[i+1,i] = -0.5*sfac
      cfac = 1.0 #2.0*kappa0
      nfac = numpy.power(cfac*numpy.sqrt(numpy.linalg.det(tij)),1.0/n)/ \
             numpy.sqrt(numpy.pi)
      val = numpy.linalg.det(tij)
      print 'n=',n,'a=',a,'detM=',val,'ana=',(n+1.0)/2**(n+1)*(numpy.exp(kappa0*L)-numpy.exp(-kappa0*L))/(kappa0*L),'nfac=',nfac
      
      wfac = 2.0*(1.0+0.5*kappa0**2*a**2)
      tij = numpy.diag([1.0]*n)
      for i in range(n-1):
         tij[i,i+1] = tij[i+1,i] = -0.5*sfac
     
      nlst.append(n)
      vlst.append(val)
   plt.plot(nlst,vlst,'ro-')
   plt.show()
   return 0

if __name__ == '__main__':
   test()
