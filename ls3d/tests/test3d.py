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
   pass 
