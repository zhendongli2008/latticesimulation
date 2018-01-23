import numpy
import scipy.linalg
import contraction2d
import matplotlib.pyplot as plt
import exact2d

def initialization(n,mass2=1.0,ng=2,iprt=0,auxbond=20):
   # Generation
   alpha = 2.0+mass2/2.0
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   xts = xts/numpy.sqrt(alpha)
   wts = wts/numpy.sqrt(alpha)
   if iprt>0:
      print '\n[initialization]'
      print ' mass2 =',mass2
      print ' alpha =',alpha
      print ' xts =',xts
      print ' wts =',wts
   # Wij = V[i,k]e[k]V[j,k] = A[i,k]A[j,k] 
   wij = numpy.zeros((ng,ng))
   for i in range(ng):
      for j in range(ng):
         wij[i,j] = numpy.exp(xts[i]*xts[j])*\
	 	    numpy.power(wts[i]*wts[j],0.25) # Absorb 1/4 for each bond(i,j)
   eig,v = scipy.linalg.eigh(wij)
   eig[numpy.argwhere(eig<0.0)] = 0.0
   wka = numpy.einsum('ik,k->ik',v,numpy.sqrt(eig))
   # Construct Z=tr(T) 
   # Shape:
   #  (2,0) (2,1) (2,2)
   #  (1,0) (1,1) (1,2)
   #  (0,0) (0,1) (0,2) . . .
   # Ordering: ludr
   zpeps = numpy.empty((n,n), dtype=numpy.object)
   # Interior
   tmp = numpy.einsum('kl,ku,kd,kr->ludr',wka,wka,wka,wka)
   for i in range(1,n-1):
      for j in range(1,n-1):	   
	 zpeps[i,j] = tmp.copy() 
   # Corners 
   tmp = numpy.einsum('k,ka,kb->ab',numpy.sqrt(wts),wka,wka)
   zpeps[0,0]     = tmp.reshape((1,ng,1,ng))
   zpeps[0,n-1]   = tmp.reshape((ng,ng,1,1))
   zpeps[n-1,0]   = tmp.reshape((1,1,ng,ng))
   zpeps[n-1,n-1] = tmp.reshape((ng,1,ng,1))
   # Boundries
   tmp = numpy.einsum('k,ka,kb,kc->abc',numpy.power(wts,0.25),wka,wka,wka)
   for j in range(1,n-1):
      zpeps[0,j]   = tmp.reshape((ng,ng,1,ng)) # bottom
      zpeps[j,0]   = tmp.reshape((1,ng,ng,ng)) # left
      zpeps[n-1,j] = tmp.reshape((ng,1,ng,ng)) # top
      zpeps[j,n-1] = tmp.reshape((ng,ng,ng,1)) # right
   # Compute scaling factor
   scale,z = contraction2d.binarySearch(zpeps,auxbond,iprt=iprt)
   # For simplicity, we assume measurement is always 
   # taken for the interior points.
   local1 = scale*numpy.einsum('k,kl,ku,kd,kr->ludr',xts,wka,wka,wka,wka)
   local2 = scale*numpy.einsum('k,kl,ku,kd,kr->ludr',xts**2,wka,wka,wka,wka)
   zpeps = scale*zpeps
   return zpeps,local1,local2

def correlationFunctions(n,mass2=1.0,ng=3,palst=None,pblst=None,iprt=0,auxbond=20):
   zpeps,local1,local2 = initialization(n,mass2,ng,iprt,auxbond)
   na = len(palst)
   nb = len(pblst) 
   cab = numpy.zeros((na,nb))
   for ia in range(na):
      for ib in range(nb):
  	 pa = palst[ia]
	 pb = pblst[ib]
         assert (pa[0]%(n-1))*(pa[1]%(n-1)) > 0
         assert (pb[0]%(n-1))*(pb[1]%(n-1)) > 0
         epeps = zpeps.copy()
         if pa == pb:
            epeps[pa] = local2.copy() 
         else:
            epeps[pa] = local1.copy() 
            epeps[pb] = local1.copy()
	 cab[ia,ib] = contraction2d.contract(epeps,auxbond)
   return cab

# Test
def test():
   m = 10
   n = 2*m+1
   mass = 0.1
   mass2 = mass**2
   print '(m,n)=',(m,n),'mass=',mass
   # Exact
   t2d = exact2d.genT2d(n,mass)
   t2d = t2d.reshape((n*n,n*n))
   tinv = scipy.linalg.inv(t2d)
   tinv = tinv.reshape((n,n,n,n))
   vpot = tinv[m,m]
   posj = range(m,m+3)
   plt.plot(posj,vpot[posj,posj],'ro-',label='exact')
   # Approximate
   for ng in [2,3,4]:
      print '\nng=',ng
      vapp = correlationFunctions(n,mass2=mass2,ng=ng,\
		      		  palst=[(m,m)],pblst=[(m,m+i) for i in range(3)])
      print vapp[0]
      plt.plot(posj,vapp[0],'bo-',label='approx (ng='+str(ng)+')')
   # Comparison
   plt.legend()
   plt.show()
   return 0


if __name__ == '__main__':
   test()
