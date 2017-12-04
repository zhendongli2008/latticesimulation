import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from zmpo_dmrg.source.mpsmpo.mpo_class import class_mpo
from zmpo_dmrg.source.mpsmpo.mps_class import class_mps

Aref = 1.0 
kappa0 = 0.01 #0#10 #0 #0

# 1D: A*exp(-kappa*x) modified from the exponential exp(-a*x)/(2a).
def genEVmpo(n,a,ng=40):
   # Quadrature
   sfac = 1.0/(1.0+0.5*kappa0**2*a**2)
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   wij = numpy.exp(sfac*numpy.einsum('i,j->ij',xts,xts))
   wij = numpy.einsum('i,ij,j->ij',numpy.sqrt(wts),wij,numpy.sqrt(wts))
   # Wij = V[i,k]e[k]V[j,k] = A[i,k]A[j,k] => BAD for large ng !!!
   eig,v = scipy.linalg.eigh(wij)
   print 'n/a/ng=',(n,a,ng)
   print eig
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
   A0 = nfac*numpy.einsum('k,ka->a',numpy.sqrt(wts),wka)
   B0 = nfac*numpy.einsum('k,ka->a',numpy.sqrt(wts)*xts,wka)*numpy.sqrt(sfac*a)
   D0 = nfac*numpy.einsum('k,ka->a',numpy.sqrt(wts)*xts*xts,wka)*(sfac*a)
   A1 = nfac*numpy.einsum('ka,kb->ab',wka,wka)
   B1 = nfac*numpy.einsum('k,ka,kb->ab',xts,wka,wka)*numpy.sqrt(sfac*a)
   D1 = nfac*numpy.einsum('k,ka,kb->ab',xts*xts,wka,wka)*(sfac*a)
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


# 1D: A*exp(-kappa*x) modified from the exponential exp(-a*x)/(2a).
def genEVmpoTaylor(n,a,ng=40,iop=0):
   # Quadrature
   sfac = 1.0/(1.0+0.5*kappa0**2*a**2)
   # OBC
   tij = numpy.diag([1.0]*n)
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -0.5*sfac
   cfac = Aref*2.0*kappa0
   nfac = numpy.power(cfac*numpy.sqrt(numpy.linalg.det(tij)),1.0/n)/ \
          numpy.sqrt(numpy.pi)
  
   import math
   def denorm(n1,n2=None):
      if n2 == None:
         return 1.0/math.sqrt(float(math.factorial(n1)))
      else:
         return 1.0/math.sqrt(float(math.factorial(n1))*float(math.factorial(n2)))


   A0 = numpy.array([0.5*(1+(-1)**n1)*sfac**(n1/2.0)*math.gamma((n1+1.0)/2.0)\
	      *denorm(n1) for n1 in range(ng)])
   A1 = numpy.array([[0.5*(1+(-1)**(n1+n2))*sfac**((n1+n2)/2.0)*math.gamma((n1+n2+1.0)/2.0)\
	      *denorm(n1,n2) for n1 in range(ng)] for n2 in range(ng)])
   nfac1 = numpy.power(cfac/reduce(numpy.dot,[A0]+[A1]*(n-2)+[A0.T]),1.0/n)
   print 'n=',n,reduce(numpy.dot,[A0]+[A1]*(n-2)+[A0.T]),\
		numpy.pi**(n/2.0)/numpy.sqrt(numpy.linalg.det(tij)),\
		nfac,nfac1
   # New normalization factor
   if iop == 1: nfac = nfac1

   # A,B,D
   A0 = nfac*numpy.array([0.5*(1+(-1)**n1)*sfac**(n1/2.0)*math.gamma((n1+1.0)/2.0)\
		   *denorm(n1) for n1 in range(ng)])
   B0 = nfac*numpy.array([0.5*(1-(-1)**n1)*sfac**(n1/2.0)*math.gamma((n1+2.0)/2.0)\
		   *denorm(n1) for n1 in range(ng)])*numpy.sqrt(sfac*a)
   D0 = nfac*numpy.array([0.5*(1+(-1)**n1)*sfac**(n1/2.0)*math.gamma((n1+3.0)/2.0)\
		   *denorm(n1) for n1 in range(ng)])*(sfac*a)
   A1 = nfac*numpy.array([[0.5*(1+(-1)**(n1+n2))*sfac**((n1+n2)/2.0)*math.gamma((n1+n2+1.0)/2.0)\
		   *denorm(n1,n2) for n1 in range(ng)] for n2 in range(ng)])
   B1 = nfac*numpy.array([[0.5*(1-(-1)**(n1+n2))*sfac**((n1+n2)/2.0)*math.gamma((n1+n2+2.0)/2.0)\
		   *denorm(n1,n2) for n1 in range(ng)] for n2 in range(ng)])*numpy.sqrt(sfac*a)
   D1 = nfac*numpy.array([[0.5*(1+(-1)**(n1+n2))*sfac**((n1+n2)/2.0)*math.gamma((n1+n2+3.0)/2.0)\
		   *denorm(n1,n2) for n1 in range(ng)] for n2 in range(ng)])*(sfac*a)

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


def test():
   # (K^-1)ij
   L = 40
   nlst = []
   kii = []
   for nc in range(2,500,10):
      n = 2*nc+1
      nlst.append(n)
      a = L/float(n+1)
      sfac = 1.0/(1.0+0.5*kappa0**2*a**2)
      tij = numpy.diag([1.0]*n)
      for i in range(n-1):
         tij[i,i+1] = tij[i+1,i] = -0.5*sfac
      vii = numpy.diag(scipy.linalg.inv(tij)) #*a*sfac*kappa0 #*(a*sfac*kappa0)
      kii.append(vii[nc])
   nlst = numpy.array(nlst)
   kii = numpy.array(kii)
   plt.plot(nlst,kii,'ko-',label='kii')
   #plt.ylim(0,3)

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
   ng = 10
   kvii = []
   kvii2 = []
   for nc in range(2,500,10):
      n = 2*nc+1
      a = L/float(n+1)
      sfac = 1.0/(1.0+0.5*kappa0**2*a**2)
      #vmpo = genEVmpoTaylor(n,a,ng)
      #kvii.append(genVij(nc,nc)/a)
      vmpo = genEVmpoTaylor(n,a,ng,iop=1)
      kvii2.append(genVij(nc,nc)/(a*sfac*kappa0))
      vmpo = genEVmpo(n,a,ng)
      kvii.append(genVij(nc,nc)/(a*sfac*kappa0))
   kvii = numpy.array(kvii)
   kvii2 = numpy.array(kvii2)
   plt.plot(nlst,kvii ,'r+--',label='Ng0='+str(ng))
   plt.plot(nlst,kvii2,'b+--',label='Ng1='+str(ng))
   plt.plot(nlst,nlst ,'g-')
   plt.legend(loc=2)
   plt.show()
   return 0

if __name__ == '__main__':
   test()
