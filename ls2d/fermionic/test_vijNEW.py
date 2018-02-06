import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from zmpo_dmrg.source.mpsmpo.mpo_class import class_mpo
from zmpo_dmrg.source.mpsmpo.mps_class import class_mps

# Boundary exponents
def kbd(ka):
   return 1.0+0.5*ka**2-ka*numpy.sqrt(1+0.25*ka**2)

# OBC
def getVij(n,a,kappa0,ifbd=False):
   tij = numpy.diag([2.0+(kappa0*a)**2]*n)
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -1.0
   if ifbd:
      bd = kbd(kappa0*a)
      tij[0,0] -= bd 
      tij[n-1,n-1] -= bd
   vij = numpy.linalg.inv(tij)*(2*kappa0*a)
   return vij

# For pairwise interactions
def genGMPO(n,left,right,middle,ifnormalize=False):
   A0,B0,D0 = left
   A2,C2,D2 = right
   A1,B1,C1,D1 = middle
   ng = A0.shape[0]
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
   site1[:ng,0]     = numpy.einsum('a,mn->amn',D2,nud)
   site1[ng:2*ng,0] = numpy.einsum('a,mn->amn',C2,nii)
   site1[2*ng:,0]   = numpy.einsum('a,mn->amn',A2,idn)
   # centeral
   # [A1,B1,D1]
   # [ 0,A1,C1]
   # [ 0, 0,A1]
   site2 = numpy.zeros((3*ng,3*ng,4,4))
   site2[:ng,:ng] 	  = numpy.einsum('ab,mn->abmn',A1,idn)
   site2[ng:2*ng,ng:2*ng] = numpy.einsum('ab,mn->abmn',A1,idn)
   site2[2*ng:,2*ng:] 	  = numpy.einsum('ab,mn->abmn',A1,idn)
   site2[:ng,ng:2*ng] 	  = numpy.einsum('ab,mn->abmn',B1,nii)
   site2[ng:2*ng,2*ng:]   = numpy.einsum('ab,mn->abmn',C1,nii)
   site2[:ng,2*ng:] 	  = numpy.einsum('ab,mn->abmn',D1,nud)
   sites = [site0]+[site2]*(n-2)+[site1]
   # Normalization
   if ifnormalize:
      Z0 = reduce(numpy.dot,[A0]+[A1]*(n-2)+[A2])
      fac = numpy.power(Z0,-1.0/n)
      print ' Z0=',Z0,' n=',n,' fac=',fac
      sites = map(lambda x:x*fac,sites)
   vmpo = class_mpo(n,sites)
   return vmpo

# Adaptive
def genEVmpoScaled(n,a,kappa0,ng=40):  
   print '\n[genEVmpoScaled]'
   print ' (n,a,kappa,ng) =',(n,a,kappa0,ng)
   # Normalization
   xi = kappa0*a  
   w = 2.0+xi**2
   lp = 0.5*(w+numpy.sqrt(w**2-4.0))
   ln = 0.5*(w-numpy.sqrt(w**2-4.0))
   nfac = numpy.sqrt(lp/(2.0*numpy.pi))*\
	  numpy.power(2.0*kappa0*numpy.sqrt(1.0-ln/lp),1.0/n)
   print ' lp,ln,nfac =',lp,ln,nfac
   # Generate quadrature
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   # Improved scaling
   alpha = 1.0 
   xts = xts/numpy.sqrt(alpha)
   wts = wts/numpy.sqrt(alpha)
   print ' alpha =',alpha
   print ' xts =',xts
   print ' wts =',wts
   # Wij = V[i,k]e[k]V[j,k] = A[i,k]A[j,k] 
   wij = numpy.zeros((ng,ng))
   for i in range(ng):
      for j in range(ng):
	 wij[i,j] = numpy.sqrt(wts[i]*wts[j])*\
		    numpy.exp( xts[i]*xts[j]\
		    +0.5*xts[i]**2*(-0.5*w+alpha)
		    +0.5*xts[j]**2*(-0.5*w+alpha))
   eig,v = scipy.linalg.eigh(wij)
   eig[numpy.argwhere(eig<0.0)] = 0.0
   print 'eig=',eig
   wka = numpy.einsum('ik,k->ik',v,numpy.sqrt(eig))
   # Boundary terms: exp(+1/2*x1**2*b1)*exp(+1/2*xN**2*bN)
   b0 = kbd(xi)
   bt = numpy.sqrt(wts)*numpy.exp(0.5*xts**2*(-0.5*w+alpha+b0))
   # A,B,D
   fac = a
   A0 = nfac*numpy.einsum('k,ka->a',bt,wka)
   B0 = nfac*numpy.einsum('k,ka->a',bt*xts,wka)*numpy.sqrt(fac)
   D0 = nfac*numpy.einsum('k,ka->a',bt*xts*xts,wka)*(fac)
   A1 = nfac*numpy.einsum('ka,kb->ab',wka,wka)
   B1 = nfac*numpy.einsum('k,ka,kb->ab',xts,wka,wka)*numpy.sqrt(fac)
   D1 = nfac*numpy.einsum('k,ka,kb->ab',xts*xts,wka,wka)*(fac)
   vmpo = genGMPO(n,[A0,B0,D0],[A0,B0,D0],[A1,B1,B1,D1])
   return vmpo

# Spinless Pseudofermion
def genFMPO(n,a,kappa0,ifbd=False):
   print '\n[genFMPO] using spinless pseudofermion'
   # A,B,D
   lam = 2.0+(kappa0*a)**2
   A1 = numpy.array([[lam ,0.,0.,-1.],
	   	     [0.  ,1.,0., 0.],
		     [0.  ,0.,1., 0.],
		     [1.  ,0.,0., 0.]])
   B1 = numpy.array([[0.,0.,-1.,0.],
	   	     [1.,0., 0.,0.],
		     [0.,0., 0.,0.],
		     [0.,0., 0.,0.]])
   C1 = numpy.array([[0.,1.,0.,0.],
	   	     [0.,0.,0.,0.],
		     [1.,0.,0.,0.],
		     [0.,0.,0.,0.]])
   D1 = numpy.array([[-1.,0.,0.,0.],
	   	     [ 0.,0.,0.,0.],
		     [ 0.,0.,0.,0.],
		     [ 0.,0.,0.,0.]])
   A0 = A1[0,:].copy()
   B0 = B1[0,:].copy()
   D0 = D1[0,:].copy()
   A2 = A1[:,0].copy()
   C2 = C1[:,0].copy()
   D2 = D1[:,0].copy()
   if ifbd: 
      lam -= kbd(kappa0*a)
      A0[0] = lam
      A2[0] = lam
   vmpo = genGMPO(n,[A0,B0,D0],[A2,C2,D2],[A1,B1,C1,D1],ifnormalize=True)
   return vmpo

def testVmpo():
   # V = 1/2*ni*Vij*nj
   def genVij(i,j):
      occ = numpy.zeros(2*n)
      if i == j:
         occ[2*i] = 1.0
         occ[2*i+1] = 1.0
      else:
         occ[2*i] = 1.0
         occ[2*j] = 1.0
      mps = class_mps(2*n)
      mps.hfstate(2*n,occ)
      mps = mps.merge([[2*i,2*i+1] for i in range(n)])
      val = mps.dot(vmpo.dotMPS(mps))
      return val
   ################
   kappa0 = 0.1
   L = 50
   nc = 40 
   ng = 10 
   ################
   n = 2*nc+1
   a = float(L)/(n+1)
   print 'a=',a
   #========================================================
   # Exact exponential
   x = numpy.arange(-nc,nc+1)*a
   z = numpy.exp(-kappa0*abs(x))
   plt.plot(x,z,'g-',label='Exact_Exp')
   
   ifbd = True 
   # Original
   vmpo = genEVmpoScaled(n,a,kappa0,ng)
   vij = numpy.array([genVij(nc,i) for i in range(n)])
   plt.plot(x,vij,'ko-',label='Num_OBC (ng='+str(ng)+')')
   
   # Vij
   for np in [nc,0,1,nc/2]:
      vij = getVij(n,a,kappa0,ifbd=ifbd)
      plt.plot(x,vij[np] ,'bx--',label='Kinv (np='+str(np)+')')
      # Fermionic
      vmpo = genFMPO(n,a,kappa0,ifbd=ifbd)
      vij = numpy.array([genVij(np,i) for i in range(n)])*(-2*kappa0*a)
      plt.plot(x,vij,'ro',label='FMPO_OBC (np='+str(np)+')')
 
   #========================================================
   #plt.ylim([0.0,1.5])
   plt.legend()
   plt.show()
   return 0

if __name__ == '__main__':
   testVmpo()
