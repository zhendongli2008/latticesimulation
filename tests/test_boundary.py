import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from zmpo_dmrg.source.mpsmpo.mpo_class import class_mpo
from zmpo_dmrg.source.mpsmpo.mps_class import class_mps

Aref = 1.0 
kappa0 = 0.0 #0#10 #0 #0

def test():
   # (K^-1)ij
   L = 10
   nlst = []
   kiib = []
   kiic = []
   for nc in range(2,1000,50):
      n = 2*nc+1
      nlst.append(n)
      a = L/float(n+1)
      sfac = 1.0/(1.0+0.5*kappa0**2*a**2)
      tij = numpy.diag([1.0]*n)
      for i in range(n-1):
         tij[i,i+1] = tij[i+1,i] = -0.5*sfac
      vii = numpy.diag(scipy.linalg.inv(tij)) #*a*sfac*kappa0 #*(a*sfac*kappa0)
      kiib.append(vii[0])
      kiic.append(vii[nc])
   nlst = numpy.array(nlst)
   kiib = numpy.array(kiib)
   kiic = numpy.array(kiic)
   plt.plot(nlst,kiib,'ro-',label='kiib')
   #plt.plot(nlst,kiic,'bo-',label='kiic')
   plt.show()
   return 0

if __name__ == '__main__':
   test()
