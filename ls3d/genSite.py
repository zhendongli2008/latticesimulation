#
# Generate the site tensor for fermionic construction 
# in 1D, 2D, and 3D by Grassmann integrations.
#
import numpy

# [coeff,op]
alpha = [[1.0,['id']],[1.0,['cbar']],[1.0,['c']],[1.0,['cbar','c']]]
beta  = [[1.0,['id']],[1.0,['c']],[-1.0,['cbar']],[-1.0,['cbar','c']]]
gamma = alpha
delta = beta
sigma = alpha
tau   = beta
plst  = [0,1,1,0]

# Product of a list of operators
def product(oplst):
   n = len(oplst)
   fac = 1.0
   op = [] 
   for i in range(n):
      fac = fac*oplst[i][0]
      op += oplst[i][1]
   return fac,op

# Remove vanishing case by simple counting no. of cbar and c
def screening(op):
   if op.count('cbar')>1: return None
   if op.count('c')>1: return None
   sop = [o for o in op if o != 'id']
   if len(sop) == 1: return None # single c and cbar
   return sop

# For partition function Z1d
def genZSite1D(lam,iop):
   print '\n[genSite.genZSite1D] iop=',iop
   zsite = numpy.zeros((4,4))
   idx = 0
   for l in range(4):
      for r in range(4):
	 if iop == 0:
            oplst = [beta[l],alpha[r]]
	 elif iop == 1:
            oplst = [beta[l],alpha[r],[1.0,['c','cbar']]]
	 elif iop == 2:
            oplst = [beta[l],[1.0,['c']],alpha[r]]
	 elif iop == 3:   
            oplst = [beta[l],[1.0,['cbar']],alpha[r]]
         fac,op = product(oplst)
         # Remove single 
	 sop = screening(op)
         if sop != None:
	    idx += 1	  
	    parity = (plst[l]+plst[r])%2
            # Do the integration with projection (1-lambda*cbar*c)
	    if len(sop) == 0:
	       assert abs(fac-1.0)<1.e-10
	       zsite[l,r] = lam
	    else:
	       if sop == ['c','cbar']:
	          zsite[l,r] = fac
	       if sop == ['cbar','c']:
	          zsite[l,r] = -fac
	    print ' idx=',idx,' l,r=',(l,r),\
		  ' fac=',fac,' sop=',sop,' parity=',parity,' zsite=',zsite[l,r]
   return zsite

# For partition function Z2d
def genZSite2D(lam,iop):
   print '\n[genSite.genZSite2D] iop=',iop
   zsite = numpy.zeros((4,4,4,4))
   idx = 0
   for l in range(4):
    for u in range(4):
     for d in range(4):
      for r in range(4):
	 if iop == 0:
            oplst = [delta[d],beta[l],gamma[u],alpha[r]]
	 elif iop == 1:
            oplst = [delta[d],beta[l],gamma[u],alpha[r],[1.0,['c','cbar']]]
	 elif iop == 2:
            oplst = [delta[d],beta[l],[1.0,['c']],gamma[u],alpha[r]]
	 elif iop == 3:   
            oplst = [delta[d],beta[l],[1.0,['cbar']],gamma[u],alpha[r]]
         fac,op = product(oplst)
         # Remove single 
	 sop = screening(op)
         if sop != None:
	    idx += 1	  
	    parity = (plst[u]+plst[l]+plst[d]+plst[r])%2
            # Do the integration with projection (1-lambda*cbar*c)
	    if len(sop) == 0:
	       assert abs(fac-1.0)<1.e-10
	       zsite[l,u,d,r] = lam
	    else:
	       if sop == ['c','cbar']:
	          zsite[l,u,d,r] = fac
	       elif sop == ['cbar','c']:
	          zsite[l,u,d,r] = -fac
	       else:
	 	  print 'no such case!'
		  exit(1)
	    print ' idx=',idx,' l,u,d,r=',(l,u,d,r),\
		  ' fac=',fac,' sop=',sop,' parity=',parity,' zsite=',zsite[l,u,d,r]
   return zsite

# For partition function Z3d
def genZSite3D(lam,iop):
   print '\n[genSite.genZSite3D] iop=',iop
   zsite = numpy.zeros((4,4,4,4,4,4)) # ludrbt
   idx = 0
   for l in range(4):
    for u in range(4):
     for d in range(4):
      for r in range(4):
       for b in range(4):
        for t in range(4):
	 if iop == 0:
            oplst = [delta[d],tau[b],beta[l],gamma[u],sigma[t],alpha[r]]
	 elif iop == 1:                                        
            oplst = [delta[d],tau[b],beta[l],gamma[u],sigma[t],alpha[r],[1.0,['c','cbar']]]
	 elif iop == 2:
            oplst = [delta[d],tau[b],beta[l],[1.0,['c']],gamma[u],sigma[t],alpha[r]]
	 elif iop == 3:   
            oplst = [delta[d],tau[b],beta[l],[1.0,['cbar']],gamma[u],sigma[t],alpha[r]]
         fac,op = product(oplst)
         # Remove single 
	 sop = screening(op)
         if sop != None:
	    idx += 1	  
	    parity = (plst[u]+plst[l]+plst[d]+plst[r]+plst[t]+plst[b])%2
            # Do the integration with projection (1-lambda*cbar*c)
	    if len(sop) == 0:
	       assert abs(fac-1.0)<1.e-10
	       zsite[l,u,d,r,b,t] = lam
	    else:
	       if sop == ['c','cbar']:
	          zsite[l,u,d,r,b,t] = fac
	       elif sop == ['cbar','c']:
	          zsite[l,u,d,r,b,t] = -fac
	       else:
	 	  print 'no such case!'
		  exit(1)
	    print ' idx=',idx,' l,u,d,r,b,t=',(l,u,d,r,b,t),\
		  ' fac=',fac,' sop=',sop,' parity=',parity,' zsite=',zsite[l,u,d,r,b,t]
   return zsite

if __name__ == '__main__':
   print '\nGenerate nonzero elements for 1D:'
   lam = 99.99
   tmp1 = genZSite1D(lam,0)
   tmp2 = numpy.array([[lam ,0.,0.,-1.],
	   	     [0.  ,1.,0., 0.],
		     [0.  ,0.,1., 0.],
		     [1.  ,0.,0., 0.]])
   print '\ndiff_A1=',numpy.linalg.norm(tmp1-tmp2)
   tmp1 = genZSite1D(lam,1)
   tmp2 = numpy.array([[1.,0.,0.,0.],
	   	     [ 0.,0.,0.,0.],
		     [ 0.,0.,0.,0.],
		     [ 0.,0.,0.,0.]])
   print '\ndiff_D1=',numpy.linalg.norm(tmp1-tmp2)
   tmp1 = genZSite1D(lam,2)
   tmp2 = numpy.array([[0.,1.,0.,0.],
	   	     [0.,0.,0.,0.],
		     [1.,0.,0.,0.],
		     [0.,0.,0.,0.]])
   print '\ndiff_B1=',numpy.linalg.norm(tmp1-tmp2)
   tmp1 = genZSite1D(lam,3)
   tmp2 = numpy.array([[0.,0.,-1.,0.],
	   	     [1.,0., 0.,0.],
		     [0.,0., 0.,0.],
		     [0.,0., 0.,0.]])
   print '\ndiff_C1=',numpy.linalg.norm(tmp1-tmp2)
   
   print '\nGenerate nonzero elements for 2D:'
   genZSite2D(lam,0)
   genZSite2D(lam,1)
   genZSite2D(lam,2)
   genZSite2D(lam,3)
   
   print '\nGenerate nonzero elements for 3D:'
   genZSite3D(lam,0)
   genZSite3D(lam,1)
   genZSite3D(lam,2)
   genZSite3D(lam,3)
