import numpy

# [coeff,op]
alpha = [[1.0,['id']],[1.0,['cbar']],[1.0,['c']],[1.0,['cbar','c']]]
beta  = [[1.0,['id']],[1.0,['c']],[-1.0,['cbar']],[-1.0,['cbar','c']]]
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

# For partition function Z
def genZSite(lam,iop):
   print '\n[nnz.genZSite] iop=',iop
   zsite = numpy.zeros((4,4,4,4))
   idx = 0
   for l in range(4):
    for u in range(4):
     for d in range(4):
      for r in range(4):
	 if iop == 0:
            oplst = [beta[u],beta[l],alpha[d],alpha[r]]
	 elif iop == 1:
            oplst = [beta[u],beta[l],alpha[d],alpha[r],[1.0,['cbar','c']]]
	 elif iop == 2:
            oplst = [beta[u],beta[l],[1.0,['cbar']],alpha[d],alpha[r]]
	 elif iop == 3:   
            oplst = [beta[u],beta[l],[1.0,['c']],alpha[d],alpha[r]]
         fac,op = product(oplst)
         # Remove single 
	 sop = screening(op)
         if sop != None:
	    idx += 1	  
	    parity = (plst[u]+plst[l]+plst[d]+plst[r])%2
	    print ' idx=',idx,' l,u,d,r=',(l,u,d,r),\
		  ' fac=',fac,' sop=',sop,' parity=',parity
            # Do the integration with projection (1-lambda*cbar*c)
	    if len(sop) == 0:
	       assert abs(fac-1.0)<1.e-10
	       zsite[l,u,d,r] = lam
	    else:
	       if sop == ['c','cbar']:
	          zsite[l,u,d,r] = fac
	       if sop == ['cbar','c']:
	          zsite[l,u,d,r] = -fac
   return zsite


if __name__ == '__main__':
   genZSite(1.0,0)
