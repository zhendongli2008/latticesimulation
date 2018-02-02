import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from zmpo_dmrg.source.mpsmpo.mpo_class import class_mpo
from zmpo_dmrg.source.mpsmpo.mps_class import class_mps

# Boundary exponents
def kbd(ka):
   return 1.0+0.5*ka**2-ka*numpy.sqrt(1+0.25*ka**2)

# OBC
def getVij(n,a,kappa0):
   tij = numpy.diag([2.0+(kappa0*a)**2]*n)
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -1.0
   bd = kbd(kappa0*a)
   tij[0,0] -= bd 
   tij[n-1,n-1] -= bd
   vij = a*numpy.linalg.inv(tij)*(2*kappa0)
   return vij

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
   alpha = 1.0 #0.28 #1.0 #xi*numpy.sqrt(1.0+0.25*xi**2)
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
      print i,j,val
      return val
   ################
   kappa0 = 0.1 #0.1 #0.001
   L = 100
   nc = 50 
   ng = 10 #2 #50
   ################
   n = 2*nc+1
   a = float(L)/(n+1)
   print 'a=',a
   #========================================================
   # Exact exponential
   x = numpy.arange(-nc,nc+1)*a
   z = numpy.exp(-kappa0*abs(x))
   plt.plot(x,z,'k-',label='Exact_Exp')
   # Vij
   vij = getVij(n,a,kappa0)
   plt.plot(x,vij[nc] ,'b+-',label='Exact_latticeInt')
   #plt.plot(x,vij[0]  ,'g+-')#,label='Exact_latticeInt')
   #plt.plot(x,vij[20] ,'r+-')#,label='Exact_latticeInt')
   # Original
   vmpo = genEVmpoScaled(n,a,kappa0,ng)
   vij = numpy.array([genVij(nc,i) for i in range(n)])+0.8
   plt.plot(x,vij,'ko-',label='Num_OBC (ng='+str(ng)+')')
   # vij = numpy.array([genVij(0,i) for i in range(n)])
   # plt.plot(x,vij,'ko--')#,label='Num_OBC (ng='+str(ng)+')')
   # vij = numpy.array([genVij(20,i) for i in range(n)])
   # plt.plot(x,vij,'ko--')#,label='Num_OBC (ng='+str(ng)+')')
   #========================================================
   plt.ylim([0.0,1.5])
   plt.legend()
   plt.show()
   return 0

if __name__ == '__main__':
   testVmpo()
