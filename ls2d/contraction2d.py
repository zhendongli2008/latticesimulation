import numpy
import mps

def binarySearch(zpeps,auxbond,maxsteps=20,erange=30,iprt=1):
   shape = zpeps.shape
   pwr = -1.0/numpy.prod(shape) 
   if iprt>0: print '\n[binarySearch] shape=',shape,'auxbond=',auxbond
   a = 0.0
   # try scale from 1/|MaxVal|
   scale = 1.0/numpy.max(map(lambda x:numpy.max(x),zpeps.flatten()))
   # *Assuming Z(1)>1.0, which is usually true? 
   b = max(2.0*scale,1.0)
   zpeps_try = zpeps*scale
   istep = 0
   while True:
      try:
         istep += 1
         z = None
         z = contract(zpeps_try,auxbond)
      except ValueError:
         pass	      
      if iprt>0: print ' istep=',istep,'(a,b)=',(a,b),'scale=',scale,'z=',z
      # Too large value of scale
      if z == None:
	 b = scale
	 scale = scale/2.0
      # Adjust scale to make z into the target region
      else:
	 if abs(z-1.0)<1.e-10:
	    if iprt>0: print ' converged scale=',scale,'z=',z
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
	    if iprt>0: print ' apply exact scale=',scale
      if istep == maxsteps: 
	 print ' binarySearch exceeds maxsteps=',maxsteps
	 exit(1)
      zpeps_try = zpeps*scale
   return scale,z

# Brute for calculations
def ratio(epeps,zpeps,auxbond=None):
   scale,z = binarySearch(zpeps,auxbond)
   epeps_try = epeps*scale
   v = contract(epeps_try,auxbond)
   print 'v,z,v/z=',v,z,v/z
   exit()
   return v/z

def contract(cpeps,auxbond=None):
   n = cpeps.shape[0]
   cmps0 = [None]*n
   # Bottom MPS
   for i in range(n):
      l,u,d,r = cpeps[0,i].shape
      assert d == 1
      cmps0[i] = numpy.reshape(cpeps[0,i], (l,u*d,r))
   # Contract
   for i in range(1,n):
      cmpo = [None]*n
      for j in range(n):
         cmpo[j] = cpeps[i,j]
      cmps0 = mpo_mapply(cmpo,cmps0)
      if auxbond is not None: # compress
         cmps0 = mps.compress(cmps0,auxbond)
   return mps.ceval(cmps0,[0]*n)

def mpo_mapply(mpo,mps):
    nsites=len(mpo)
    assert len(mps)==nsites
    ret=[None]*nsites
    if len(mps[0].shape)==3: 
        # mpo x mps
        for i in xrange(nsites):
            assert mpo[i].shape[2]==mps[i].shape[1]
            mt=numpy.einsum("apqb,cqd->acpbd",mpo[i],mps[i])
            mt=numpy.reshape(mt,[mpo[i].shape[0]*mps[i].shape[0],mpo[i].shape[1],
                             mpo[i].shape[-1]*mps[i].shape[-1]])
            ret[i]=mt
    elif len(mps[0].shape)==4: 
        # mpo x mpo
        for i in xrange(nsites):
            assert mpo[i].shape[2]==mps[i].shape[1]
            mt=numpy.einsum("apqb,cqrd->acprbd",mpo[i],mps[i])
            mt=numpy.reshape(mt,[mpo[i].shape[0]*mps[i].shape[0],
                             mpo[i].shape[1],mps[i].shape[2],
                             mpo[i].shape[-1]*mps[i].shape[-1]])
            ret[i]=mt
    return ret
