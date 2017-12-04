import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from zmpo_dmrg.source.mpsmpo.mpo_class import class_mpo
from zmpo_dmrg.source.mpsmpo.mps_class import class_mps

A = 1.0
kappa = 1.0

# Z=A*exp(-kr)
def getVij(n,a,A,kappa,Z=-1.0):
   # OBC
   tij = numpy.diag([2.0+(kappa*a)**2]*n)
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -1.0
   vij = -Z*A*a*numpy.linalg.inv(tij)*(2*kappa)
   return vij


# 1D: A*exp(-kappa*x) modified from the exponential exp(-a*x)/(2a).
def genEVmpo(n,a,ng=40):
   # Quadrature
   sfac = 1.0/(1.0+0.5*kappa**2*a**2)
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   wij = numpy.exp(sfac*numpy.einsum('i,j->ij',xts,xts))
   wij = numpy.einsum('i,ij,j->ij',numpy.sqrt(wts),wij,numpy.sqrt(wts))
   # Wij = V[i,k]e[k]V[j,k] = A[i,k]A[j,k] => BAD for large ng !!!
   eig,v = scipy.linalg.eigh(wij)
   print 'ng=',ng
   print eig
   eig[numpy.argwhere(eig<0.0)] = 0.0
   wka = numpy.einsum('ik,k->ik',v,numpy.sqrt(eig))
   # OBC
   tij = numpy.diag([1.0]*n)
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -0.5*sfac
   cfac = 2.0*kappa
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
def genEVmpoTaylor(n,a,ng=40):
   # Quadrature
   sfac = 1.0/(1.0+0.5*kappa**2*a**2)
   # OBC
   tij = numpy.diag([1.0]*n)
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -0.5*sfac
   cfac = 2.0*kappa
   nfac = numpy.power(cfac*numpy.sqrt(numpy.linalg.det(tij)),1.0/n)/ \
          numpy.sqrt(numpy.pi)
   
   import math
   def denorm(n1,n2=None):
      if n2 == None:
         return 1.0/math.sqrt(float(math.factorial(n1)))
      else:
         return 1.0/math.sqrt(float(math.factorial(n1))*float(math.factorial(n2)))

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
   # V = 1/2*ni*Vij*nj
   def genVij(vmpo,i,j):
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

   L = 10 
   for ng,cl in zip([10,20,30,50,80,110],['r','b','g','y','c','k']):
      err1 = []
      err2 = []
      nlst = []
      for nc in range(5,100,5):
         n = 2*nc+1
         nlst.append(n)
         a = L/float(n)
         # ExactValue
         vii = numpy.diag(getVij(n,a,A,kappa))
         x = numpy.arange(-nc,nc+1)*a
         # Numerical
         vmpo = genEVmpoTaylor(n,a,ng)
         vij_mid = genVij(vmpo,nc,nc)
         diff = abs(vij_mid-vii[nc])
         err1.append(diff)
         # Gauss
	 vmpo = genEVmpo(n,a,ng)
         vij_mid = genVij(vmpo,nc,nc)
         diff = abs(vij_mid-vii[nc])
         err2.append(diff)
      # test
      plt.plot(nlst,err1,cl+'o-',label=str(ng))
      plt.plot(nlst,err2,cl+'o--')#,label=str(ng))
      print 'nlst=',nlst
      print 'err1=',err1
      print 'err2=',err2
   plt.legend()
   plt.show()
   exit()

   L = 10 
   for nc,cl in zip([10,20,30,40,50,60],['r','b','g','y','c','k']):
      n = 2*nc+1
      a = L/float(n)
      err1 = []
      err2 = []
      nlst = []
      for ng in [10,30,60,90,120,150]:
         nlst.append(ng)
         # ExactValue
         vii = numpy.diag(getVij(n,a,A,kappa))
         x = numpy.arange(-nc,nc+1)*a
         # Numerical
         vmpo = genEVmpoTaylor(n,a,ng)
         vij_mid = genVij(vmpo,nc,nc)
         diff = abs(vij_mid-vii[nc])
         err1.append(diff)
         # Gauss
	 vmpo = genEVmpo(n,a,ng)
         vij_mid = genVij(vmpo,nc,nc)
         diff = abs(vij_mid-vii[nc])
         err2.append(diff)
      # test
      plt.semilogy(nlst,err1,cl+'o-',label='N='+str(n))
      plt.semilogy(nlst,err2,cl+'o--')#,label=str(ng))
      print 'nlst=',nlst
      print 'err1=',err1
      print 'err2=',err2
   plt.legend()
   plt.show()
   return 0

if __name__ == '__main__':
   test()
