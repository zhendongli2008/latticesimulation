import numpy
import scipy.linalg

# Evaluates mps at given config
def ceval(mps,config):
   mps_mats=[None]*len(config)
   nsites=len(config)
   for i, pval in enumerate(config):
       mps_mats[i]=mps[i][:,pval,:]
   # multiply "backwards" from right to left
   val=mps_mats[0]
   for i in xrange(1,nsites):
       val=numpy.dot(val,mps_mats[i])
   # turn into scalar
   return numpy.trace(val)

def compress(mps,maxM):
   mps1 = leftCanon(mps,Dcut=-1)
   mps1 = rightCanon(mps1,Dcut=maxM)
   mps1 = leftCanon(mps1,Dcut=maxM)
   return mps1

# Left Canonical MPS
def leftCanon(sites,thresh=1.e-12,Dcut=-1,debug=False):
   nsite = len(sites)
   tsite = sites[0].copy()
   link = [0]*(nsite-1)
   sval = [0]*(nsite-1)
   lmps = [0]*nsite
   # rk3
   for isite in range(nsite-1):
      s = tsite.shape
      d1 = s[0]*s[1]
      d2 = s[2]
      mat = tsite.reshape((d1,d2)).copy() 
      u,sigs,vt = mps_svd_cut(mat,thresh,Dcut)
      bdim = len(sigs)
      if debug:
         print '-'*80	 
         print ' Results[i]:',isite
         print '-'*80	 
         print ' dimension:',(d1,d2),'->',bdim
         sum2 = numpy.sum(numpy.array(sigs)**2)
         dwts = 1.0-sum2
         print ' sum of sigs2:',sum2,' dwts:',dwts
         print ' sigs:\n',sigs
      sval[isite] = sigs.copy()
      lmps[isite] = u.reshape((s[0],s[1],bdim)).copy()
      tmp = numpy.diag(sigs).dot(vt)
      link[isite] = tmp.copy()
      #tsite = numpy.einsum('sl,lur->sur',tmp,sites[isite+1])
      tsite = numpy.tensordot(tmp,sites[isite+1],axes=([1],[0]))
   norm = numpy.linalg.norm(tsite)
   lmps[nsite-1] = tsite #/norm
   return lmps

# Right Canonical MPS
def rightCanon(sites,thresh=1.e-12,Dcut=-1,debug=False):
   nsite = len(sites)
   tsite = sites[nsite-1].copy()
   link = [0]*(nsite-1)
   sval = [0]*(nsite-1)
   rmps = [0]*nsite
   # rk3
   for isite in range(nsite-1,0,-1):
      s = tsite.shape
      d1 = s[0]
      d2 = s[1]*s[2]
      mat = tsite.reshape((d1,d2)).copy()
      # C=U s Vd => Ct=V* s Ut 
      u,sigs,vt = mps_svd_cut(mat.T,thresh,Dcut)
      bdim = len(sigs)
      if debug:
         print '-'*80	 
         print ' Results[i]:',isite
         print '-'*80	 
         print ' dimension:',(d1,d2),'->',bdim
         sum2 = numpy.sum(numpy.array(sigs)**2)
         dwts = 1.0-sum2
         print ' sum of sigs2:',sum2,' dwts:',dwts
         print ' sigs:\n',sigs
      sval[isite-1] = sigs.copy()
      # u = V* ---> (Vd)[alpha,n*r]
      rmps[isite] = u.T.reshape((bdim,s[1],s[2])).copy()
      tmp = numpy.diag(sigs).dot(vt).T.copy()
      link[isite-1] = tmp.copy()
      #tsite = numpy.einsum('lur,rs->lus',sites[isite-1],tmp)
      tsite = numpy.tensordot(sites[isite-1],tmp,axes=([2],[0]))
   norm = numpy.linalg.norm(tsite)
   rmps[0] = tsite #/norm
   return rmps

def mps_svd_cut(mat,thresh,D):
   if len(mat.shape) != 2:
      print "NOT A MATRIX in MPS_SVD_CUT !",mat.shape
      exit(1) 
   d1, d2 = mat.shape
   #------------------------------------
   u, sig, v = scipy.linalg.svd(mat, full_matrices=False, lapack_driver='gesvd')
   #------------------------------------
   # decide the final dimension of output according to thresh & D:
   r=len(sig)
   for i in range(r):
      if(sig[i]<thresh*1.01):
         r=i
         break
   # bond dimension at least = 1
   if r==0: r=1
   # final bond dimension:
   if D>0 :
      bdim=min(r,D) # D - use more flexible bond dimension
      rkep=min(r,D)
   else:
      bdim=r
      rkep=r
   s2=numpy.zeros((bdim))
   u2=numpy.zeros((d1,bdim),dtype = mat.dtype)
   v2=numpy.zeros((bdim,d2),dtype = mat.dtype)
   s2[:rkep]=sig[:rkep]
   u2[:,:rkep]=u[:,:rkep]
   v2[:rkep,:]=v[:rkep,:]
   return u2,s2,v2
