import numpy
import scipy.linalg
import contraction2d
import matplotlib.pyplot as plt
import exact2d

def genVpeps(n,mass2=1.0,ng=3,pa=(1,1),pb=(1,2),iprt=0,auxbond=None):
   
   # Generation
   alpha = 2.0+mass2/2.0
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   xts = xts/numpy.sqrt(alpha)
   wts = wts/numpy.sqrt(alpha)
   if iprt>0:
      print '\n[genVpeps]'
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
   print ' eig =',eig
   eig[numpy.argwhere(eig<0.0)] = 0.0
   wka = numpy.einsum('ik,k->ik',v,numpy.sqrt(eig))
   
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

   # For simplicity, we assume measurement is always 
   # taken for the interior points.
   assert (pa[0]%(n-1))*(pa[1]%(n-1)) > 0
   assert (pb[0]%(n-1))*(pb[1]%(n-1)) > 0
   epeps = zpeps.copy()
   if pa == pb:
      tmp = numpy.einsum('k,kl,ku,kd,kr->ludr',xts**2,wka,wka,wka,wka)
      epeps[pa] = tmp.copy() 
   else:
      tmp = numpy.einsum('k,kl,ku,kd,kr->ludr',xts,wka,wka,wka,wka)
      epeps[pa] = tmp.copy() 
      epeps[pb] = tmp.copy()

   # Contract
   cij = contraction2d.ratio(epeps,zpeps,auxbond=20)
   return cij


def test():
   m = 4
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
   posj = range(1,n-1)
   plt.plot(posj,vpot[posj,posj],'ro-',label='exact')
   # Approximate
   for ng in [2,4,6,8]:
      print '\nng=',ng
      vapp = []
      for j in posj:
         pb = (j,j)	   
         cij = genVpeps(n,mass2=mass2,ng=ng,pa=(m,m),pb=pb)
         print 'j=',j,'pb=',pb,'cij=',cij,'eij=',vpot[pb]
         vapp.append(cij)
      plt.plot(posj,vapp,'bo-',label='approx (ng='+str(ng)+')')
   # Comparison
   plt.legend()
   plt.show()
   return 0


if __name__ == '__main__':
   test()
