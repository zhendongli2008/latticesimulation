import numpy
from latticesimulation.ls2d import contraction2d

def binarySearch(zpeps,auxbond,maxsteps=50,erange=30,iprt=0):
   shape = zpeps.shape
   pwr = -1.0/numpy.prod(shape) 
   if iprt>0: print '\n[binarySearch] shape=',shape,'auxbond=',auxbond
   a = 0.0
   # try scale from 1/|MaxVal|
   scale = 1.0/numpy.max(map(lambda x:numpy.max(x),zpeps.flatten()))
   print ' scale0=',scale
   b = max(2.0*scale,10.0)
   zpeps_try = zpeps*scale
   istep = 0
   while True:
      try:
         istep += 1
         z = None
         z = contract(zpeps_try,auxbond)
      except ValueError:
         pass	      
      if iprt>0: print ' i= ',istep,'(a,b,w)=',(a,b,b-a),\
		       'scale=',scale,'z=',z
      # Too large value of scale
      if z == None:
	 b = scale
	 scale = (a+b)/2.0
      # Adjust scale to make z into the target region
      else:
	 if abs(z-1.0)<1.e-10:
	    if iprt>0: print ' converged scale=',scale,'z=',z
	    break
         if abs(b-a) < 1.e-10:
 	    print ' No good solution exists',scale,'z=',z
            break
	 # Update interval
	 if z > 1.0:
	    b = scale
	 else:
	    a = scale
	 # initial stage apply binary search
   	 if z < 10.0**(-erange) or z > 10.0**erange:
	    scale = (a+b)/2.0
	 # close to convergence, apply ''exact'' scale
  	 else: 
	    sfac = numpy.power(z,pwr)
	    scale = scale*sfac
	    if scale > a and scale < b:
	       if iprt>0: print ' apply "exact" scale=',scale
	    else:
	       # Reset to binary search
	       scale = (a+b)/2.0
      # Check convergence
      if istep == maxsteps: 
	 print ' binarySearch exceeds maxsteps=',maxsteps
	 break
      zpeps_try = zpeps*scale
   # Final step
   scale = scale*numpy.power(z,pwr)
   return scale,z

# BRUTE-FORCE 
def contract(cpeps3d,auxbond=None):
   l,m,n = cpeps3d.shape
   cpeps2d = numpy.empty((n,n), dtype=numpy.object) # (z,x,y) / (ludrbt)
   for i in range(m):
     for j in range(n):
       cpeps2d[i,j] = numpy.ones(1).reshape(1,1,1,1,1,1)
   # Pack
   for z in range(l):
      for i in range(m):
         for j in range(n):
	    tmp = numpy.einsum('ludrbt,LUDRBb->lLuUdDrRBt',cpeps3d[z,i,j],cpeps2d[i,j])
            s = tmp.shape
            cpeps2d[i,j] = tmp.reshape((s[0]*s[1],s[2]*s[3],s[4]*s[5],s[6]*s[7],s[8],s[9]))
   # Reform	   
   cpeps2d_new = numpy.empty((n,n), dtype=numpy.object) # (z,x,y) / (ludrbt)
   for i in range(m):
      for j in range(n):
         # ludrbt
	 s = cpeps2d[i,j].shape
	 assert s[4] == 1 and s[5] == 1
	 cpeps2d_new[i,j] = cpeps2d[i,j].reshape((s[0],s[1],s[2],s[3]))
   val = contraction2d.contract(cpeps2d_new,auxbond)
   return val
