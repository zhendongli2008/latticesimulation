import numpy
import scipy.linalg
import nnz

def initialization(n,mass2=1.0,iprt=0,auxbond=20):
   print '\n[test3d.initialization] n=',n,' mass2=',mass2
   # Construct Z=tr(T) 
   # Shape:
   #  (2,0) (2,1) (2,2)
   #  (1,0) (1,1) (1,2)
   #  (0,0) (0,1) (0,2) . . .
   # Ordering: ludrbt
   zpeps = numpy.empty((n,n,n), dtype=numpy.object) # (z,x,y)
   # Interior
   d = 4 
   lam = 6.0+mass2
   tint = nnz.genZSite3D(lam,0)
   for k in range(1,n-1):
    for i in range(1,n-1):
     for j in range(1,n-1):
      zpeps[k,i,j] = tint.copy()
   # 8 Corners - T[ludrbt]
   zpeps[0,0,0]     = tint[:1,:,:1,:, :1,:].copy()
   zpeps[0,0,n-1]   = tint[:,:,:1,:1, :1,:].copy()
   zpeps[0,n-1,0]   = tint[:1,:1,:,:, :1,:].copy()
   zpeps[0,n-1,n-1] = tint[:,:1,:,:1, :1,:].copy()
   # NEW:
   zpeps[n-1,0,0]     = tint[:1,:,:1,:, :,:1].copy()
   zpeps[n-1,0,n-1]   = tint[:,:,:1,:1, :,:1].copy()
   zpeps[n-1,n-1,0]   = tint[:1,:1,:,:, :,:1].copy()
   zpeps[n-1,n-1,n-1] = tint[:,:1,:,:1, :,:1].copy()
   # 12 Boundaries - T[ludrbt]
   for j in range(1,n-1):
      zpeps[0,0,j]   = tint[:,:,:1,:, :1,:].copy()
      zpeps[0,j,0]   = tint[:1,:,:,:, :1,:].copy()
      zpeps[0,n-1,j] = tint[:,:1,:,:, :1,:].copy()
      zpeps[0,j,n-1] = tint[:,:,:,:1, :1,:].copy()
      # NEW
      zpeps[n-1,0,j]   = tint[:,:,:1,:, :,:1].copy()
      zpeps[n-1,j,0]   = tint[:1,:,:,:, :,:1].copy()
      zpeps[n-1,n-1,j] = tint[:,:1,:,:, :,:1].copy()
      zpeps[n-1,j,n-1] = tint[:,:,:,:1, :,:1].copy()
      # NEW
      zpeps[j,0,0]     = tint[:1,:,:1,:, :,:].copy()
      zpeps[j,0,n-1]   = tint[:,:,:1,:1, :,:].copy()
      zpeps[j,n-1,0]   = tint[:1,:1,:,:, :,:].copy()
      zpeps[j,n-1,n-1] = tint[:,:1,:,:1, :,:].copy()
   # 8 faces - T(ludrbt)
   for i in range(1,n-1):
     for j in range(1,n-1):
	# bottom & top
	zpeps[0,i,j] = tint[:,:,:,:,:1,:].copy()
	zpeps[n-1,i,j] = tint[:,:,:,:,:,:1].copy()
        #  
	zpeps[i,0,j] = tint[:,:,:1,:,:,:].copy()
	zpeps[i,n-1,j] = tint[:,:1,:,:,:,:].copy()
	# 
	zpeps[i,j,0] = tint[:1,:,:,:,:,:].copy()
	zpeps[i,j,n-1] = tint[:,:,:,:1,:,:].copy()
   # CHECK
   for k in range(n):
    for i in range(n):
     for j in range(n):
       if zpeps[k,i,j] == None:
	 print (k,i,j),zpeps[k,i,j]
   # Compute scaling factor
   import contraction3d
   scale,z = contraction3d.binarySearch(zpeps,auxbond,iprt=iprt)
   # Local terms
   local2  = scale*nnz.genZSite3D(lam,1)
   local1a = scale*nnz.genZSite3D(lam,2)
   local1b = scale*nnz.genZSite3D(lam,3)
   zpeps0 = zpeps.copy()
   zpeps = scale*zpeps
   return scale,zpeps0,zpeps,local2,local1a,local1b


if __name__ == '__main__':
   #
   # NOTE: 1. we can always choose a=1 as the unit of length!
   #       then we observe how much fictious sites are required!
   #       2. for small mass, a fairly large n is required,
   #	   the estimate in latticeQED paper suggests n=1000.
   #
   # 	   This is hard to test for direct inversion, but maybe 
   #       possible by 3D contraction???
   #
   m = 1 #12
   n = 2*m+1
   mass = 1.0
   a = 1.0
   iprt = 1
   auxbond = 100
   print 'n=',n

   # Test Z:
   from latticesimulation.ls2d import exact2d
   import matplotlib.pyplot as plt
   t3d = exact2d.genT3d(n,mass*a)
   t3d = t3d.reshape((n**3,n**3))
   print t3d.shape
   tinv = scipy.linalg.inv(t3d)
   tinv = tinv.reshape((n,n,n,n,n,n))
   print '\nTest Z:'
   print 'direct detM=',scipy.linalg.det(t3d)
   
   def exactFun(x):
      return numpy.exp(-mass*x)/(4*numpy.pi*x)
   x = numpy.linspace(0.1,a*m*numpy.sqrt(3),100)
   y = exactFun(x)
   plt.plot(x,y,'k-',label='exact')
   tinv = tinv/a
   xh = numpy.arange(m+1)*a
   plt.plot(xh,tinv[m,m,m][m,m,range(m,n)],'go-',label='horizontal')
   xd = numpy.arange(m+1)*a*numpy.sqrt(3)
   plt.plot(xd,tinv[m,m,m][range(m,n),range(m,n),range(m,n)],'ro-',label='diagonal')
   plt.legend()
   plt.xlim([1,10])
   plt.ylim([0,0.1])
   #plt.show()

   # Only 2*2*2
   import contraction3d
   mass = 1.e-3
   a = 1.0
   n = 2
   auxbond = 1000
   t3d = exact2d.genT3d(n,mass*a)
   t3d = t3d.reshape((n**3,n**3))
   tinv = scipy.linalg.inv(t3d)
   tinv = tinv.reshape((n,n,n,n,n,n))
   result = initialization(n,mass**2,iprt,auxbond)
   scale,zpeps0,zpeps,local2,local1a,local1b = result
   print '\nTest Z:'
   print 'direct detM=',scipy.linalg.det(t3d)
   print 'scale=',scale,'Z=',numpy.power(1.0/scale,n*n*n)
   print 'tr(T)=',contraction3d.contract(zpeps0,auxbond)

   for i in range(n):
    for j in range(n):
     for k in range(n):
        epeps3d = zpeps.copy()
	s = epeps3d[i,j,k].shape
	epeps3d[i,j,k] = local2[:s[0],:s[1],:s[2],:s[3],:s[4],:s[5]]
        print '(i,j,k)=',(i,j,k),contraction3d.contract(epeps3d,auxbond),\
	       tinv[i,j,k,i,j,k]
