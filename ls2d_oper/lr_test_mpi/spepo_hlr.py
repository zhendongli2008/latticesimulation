import numpy as np
#import autograd.numpy as np
from latticesimulation.ls2d.opt_simple import peps
import h5py
import time
import peps_contraction
import autograd

#einsum=autograd.numpy.einsum
einsum=np.einsum

#contraction = peps.contract_cpeps 
contraction = peps_contraction.contract_cpeps

ifprint = False
dirname = '../tmp2_4by4nf4'
nf = 4
abond = 40

def loadPEPO(fname,iprt=0):
   if ifprint and iprt>0: print '[spepo_hlr.loadPEPO] fname=',fname
   f = h5py.File(fname,'r')
   m,n = f['shape'].value
   pepo = np.empty((m,n),dtype=np.object)
   for i in range(m):
      for j in range(n):
         pepo[i,j] = f['site_'+str(i)+'_'+str(j)].value 
   f.close()
   return pepo

def eval_heish(pepsa, pepsb, iop):
   # Load
   f = h5py.File(dirname+'/fitCoulomb.h5','r')
   indx = f['indx_final'].value 
   mlst = f['mlst_final'].value  
   clst = f['clst_final'].value  
   f.close()
   # Compute <PEPS|H[i]|PEPS> 
   nr,nc = pepsa.shape
   assert nr == nc
   L = nr
   val = 0.
   fac = [1.0,0.5,0.5]
   if iop != 3:
      iclst = [iop]
   else:
      iclst = [0,1,2]
   for k in range(len(indx)):
      for ic in iclst:
         t0 = time.time()
         fname = dirname+'/spepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
         spepo = loadPEPO(fname,iprt=1)
         t1 = time.time()
         tmp = evalContraction(spepo,pepsa,pepsb,abond) 
         t2 = time.time()
         val += clst[k]*fac[ic]*tmp
         if ifprint:
            print 
	    print 'ic,k=',(ic,k),'clst=',clst[k],'fac=',fac[ic],'tmp=',tmp,'val=',val
            print 'time for loading =',t1-t0,' evaluation =',t2-t1
   # Convert scale
   val = val*(nf+1.0)
   return val

def evalContraction(spepo,pepsa,pepsb,auxbond):
    epeps0 = np.empty(spepo.shape, dtype=np.object)
    L = pepsa.shape[0]
    Ltot = L+nf*(L-1)
    assert Ltot+2 == epeps0.shape[0]
    psites = [1+i*(nf+1) for i in range(L)]
    # Unaffected interior parts
    for ii in range(Ltot+2):
       for jj in range(Ltot+2):
          epeps0[ii,jj] = spepo[ii,jj][0,0]
    # Action on physical sites
    for i in range(L):
       ii = 1+i*(nf+1) 
       for j in range(L):
          jj = 1+j*(nf+1) 
	  tmp1 = einsum('pludr,pqLUDR->qlLuUdDrR',pepsa[i,j],spepo[ii,jj])
	  s = tmp1.shape
	  tmp1 = np.reshape(tmp1,(s[0],s[1]*s[2],s[3]*s[4],s[5]*s[6],s[7]*s[8]))
	  tmp1 = einsum('pludr,pLUDR->lLuUdDrR',tmp1,pepsb[i,j])
	  s = tmp1.shape
	  epeps0[ii,jj] = np.reshape(tmp1,(s[0]*s[1],s[2]*s[3],s[4]*s[5],s[6]*s[7]))
    # Intermediate rows
    D = pepsa[0,0].shape[-1]
    it = np.identity(D).reshape((D,1,1,D))
    for ii in psites:
       for jj in range(1,Ltot+1):
          if jj in psites: continue
	  tmp1 = einsum('ludr,LUDR->lLuUdDrR',it,spepo[ii,jj][0,0])
	  s = tmp1.shape
	  tmp1 = np.reshape(tmp1,(s[0]*s[1],s[2]*s[3],s[4]*s[5],s[6]*s[7]))
	  tmp1 = einsum('ludr,LUDR->lLuUdDrR',tmp1,it)
	  s = tmp1.shape
	  epeps0[ii,jj] = np.reshape(tmp1,(s[0]*s[1],s[2]*s[3],s[4]*s[5],s[6]*s[7]))
    # Intermediate columns 
    it = np.identity(D).reshape((1,D,D,1))
    for jj in psites:
       for ii in range(1,Ltot+1):
          if ii in psites: continue
	  tmp1 = einsum('ludr,LUDR->lLuUdDrR',it,spepo[ii,jj][0,0])
	  s = tmp1.shape
	  tmp1 = np.reshape(tmp1,(s[0]*s[1],s[2]*s[3],s[4]*s[5],s[6]*s[7]))
	  tmp1 = einsum('ludr,LUDR->lLuUdDrR',tmp1,it)
	  s = tmp1.shape
	  epeps0[ii,jj] = np.reshape(tmp1,(s[0]*s[1],s[2]*s[3],s[4]*s[5],s[6]*s[7]))
    # Contract 
    val = contraction(epeps0, auxbond)
    return val
