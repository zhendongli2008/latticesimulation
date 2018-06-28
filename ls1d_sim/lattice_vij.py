import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from zmpo_dmrg.source.mpsmpo.mpo_class import class_mpo
from zmpo_dmrg.source.mpsmpo.mps_class import class_mps

Aref = 1.071925
kappa0 = 1.0/2.385345 # eq(7) of prb91,235151(2015)

# A*exp(-k|x|)
def getVint(A,kappa,x):
   return A*numpy.exp(-kappa*abs(x))

def getVij(n,a,A,kappa,Z=-1.0):
   # OBC
   tij = numpy.diag([2.0+(kappa*a)**2]*n)
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -1.0
   vij = -Z*A*a*numpy.linalg.inv(tij)*(2*kappa)
   return vij

# Vne = vne(i)*ni - put atom on the center for simplicity
def getVne(n,a,Z=1.0):
   vij = getVij(n,a,Aref,kappa0,Z)
   #plt.plot(vij[nc,:],'ro-')
   #plt.show()
   #print 'vne_max=',vij[nc,nc]
   assert n%2 == 1
   nc = (n-1)/2
   vne = numpy.zeros((2*n))
   for i in range(n):
      vne[2*i] = vij[nc,i]
      vne[2*i+1] = vij[nc,i]
   return vne 

# 1D: A*exp(-kappa*x) modified from the exponential exp(-a*x)/(2a).
def genEVmpo(n,a,ng=40):
   # Quadrature
   sfac = 1.0/(1.0+0.5*kappa0**2*a**2)
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   wij = numpy.exp(sfac*numpy.einsum('i,j->ij',xts,xts))
   # Wij = V[i,k]e[k]V[j,k] = A[i,k]A[j,k] => BAD for large ng !!!
   eig,v = scipy.linalg.eigh(wij)
   eig[numpy.argwhere(eig<0.0)] = 0.0
   wka = numpy.einsum('ik,k->ik',v,numpy.sqrt(eig))
   # OBC
   tij = numpy.diag([1.0]*n)
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -0.5*sfac
   cfac = Aref*2.0*kappa0
   nfac = numpy.power(cfac*numpy.sqrt(numpy.linalg.det(tij)),1.0/n)/ \
          numpy.sqrt(numpy.pi)
   # A,B,D
   A0 = nfac*numpy.einsum('k,ka->a',wts,wka)
   B0 = nfac*numpy.einsum('k,ka->a',wts*xts,wka)*numpy.sqrt(sfac*a)
   D0 = nfac*numpy.einsum('k,ka->a',wts*xts*xts,wka)*(sfac*a)
   A1 = nfac*numpy.einsum('k,ka,kb->ab',wts,wka,wka)
   B1 = nfac*numpy.einsum('k,ka,kb->ab',wts*xts,wka,wka)*numpy.sqrt(sfac*a)
   D1 = nfac*numpy.einsum('k,ka,kb->ab',wts*xts*xts,wka,wka)*(sfac*a)
   # Construction of MPOs 
   idn = numpy.identity(4)
   nii = numpy.zeros((4,4))
   nii[1,1] = 1.0
   nii[2,2] = 1.0
   nud = numpy.zeros((4,4))
   nud[3,3] = 1.0
   # first [A0,B0,D0]
   site0 = numpy.zeros((1,3*ng,4,4))
   site0[0,:ng]     = numpy.einsum('a,mn->amn',A0,idn)
   site0[0,ng:2*ng] = numpy.einsum('a,mn->amn',B0,nii)
   site0[0,2*ng:]   = numpy.einsum('a,mn->amn',D0,nud)
   # last [D0,B0,A0]
   site1 = numpy.zeros((3*ng,1,4,4))
   site1[:ng,0]     = numpy.einsum('a,mn->amn',D0,nud)
   site1[ng:2*ng,0] = numpy.einsum('a,mn->amn',B0,nii)
   site1[2*ng:,0]   = numpy.einsum('a,mn->amn',A0,idn)
   # centeral
   # [A1,B1,D1]
   # [ 0,A1,B1]
   # [ 0, 0,A1]
   site2 = numpy.zeros((3*ng,3*ng,4,4))
   site2[:ng,:ng] 	  = numpy.einsum('ab,mn->abmn',A1,idn)
   site2[ng:2*ng,ng:2*ng] = numpy.einsum('ab,mn->abmn',A1,idn)
   site2[2*ng:,2*ng:] 	  = numpy.einsum('ab,mn->abmn',A1,idn)
   site2[:ng,ng:2*ng] 	  = numpy.einsum('ab,mn->abmn',B1,nii)
   site2[ng:2*ng,2*ng:]   = numpy.einsum('ab,mn->abmn',B1,nii)
   site2[:ng,2*ng:] 	  = numpy.einsum('ab,mn->abmn',D1,nud)
   sites = [site0]+[site2]*(n-2)+[site1]
   vmpo = class_mpo(n,sites)
   return vmpo

def testVmpo(n,a,ng=20):
   vmpo = genEVmpo(n,a,ng)
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
   # Plot
   assert n%2 == 1
   nc = (n-1)/2
   # 
   # Exact exponential
   # 
   x = numpy.arange(-nc,nc+1)*a
   y = Aref*numpy.exp(-kappa0*abs(x))
   plt.plot(x,y,'ko-',label='Exact_Exp')
   plt.xlim(x[0],x[-1])
   #
   # Exact OBC: [0]_1_2_3_|_5_6_7_[8] nsite=7
   #
   L = (n+1)*a
   z = Aref*2.0*numpy.sinh(kappa0*(x+L/2))*\
       numpy.sinh(kappa0*(L/2-x))/\
       (numpy.sinh(kappa0*L))
   plt.plot(x,z,'b--',label='Exact_OBC')
   #
   # Inverse
   # 
   vne = getVne(n,a,Z=-1.0)
   plt.plot(x,vne[::2],'g+-',label='Inv_OBC',markersize=15)
   #
   # Numerical
   # 
   vij_mid = []
   for i in range(n):
      vij_mid.append(genVij(int(n/2),i))
   vij_mid = numpy.array(vij_mid)
   plt.plot(x,vij_mid,'ro-',label='Num_OBC')
   plt.legend()
   plt.show()
   return 0

def testParams():  
   ng = 10
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   plt.semilogy(xts,wts,'ko',label='ng=10')
   ng = 20
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   plt.semilogy(xts,wts,'bo',label='ng=50')
   ng = 30
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   plt.semilogy(xts,wts,'ro',label='ng=30')
   ng = 40
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   plt.semilogy(xts,wts,'go',label='ng=40')
   ng = 50
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   plt.semilogy(xts,wts,'co',label='ng=50')
   plt.legend()
   plt.show()
   # OBC
   a = 0.1
   sfac = 1.0/(1.0+0.5*kappa0**2*a**2)
   for n in [10,100,500,1000]:
      tij = numpy.diag([1.0]*n)
      for i in range(n-1):
         tij[i,i+1] = tij[i+1,i] = -0.5*sfac
      cfac = Aref*2.0*kappa0
      nfac = numpy.power(cfac*numpy.sqrt(numpy.linalg.det(tij)),1.0/n)/ \
             numpy.sqrt(numpy.pi)
      print 'n=',n,'nfac=',nfac
   return 0

if __name__ == '__main__':
   n = 101
   a = 0.2
   testVmpo(n,a,ng=24)
