import autograd
import autograd.numpy as np
from latticesimulation.ls2d.opt_simple import peps
import h5py
einsum=autograd.numpy.einsum

dirname = '../tmp2'
nf = 4
abond = 20

def loadPEPO(fname,iprt=0):
   if iprt>0: print '[spepo_hlr.loadPEPO] fname=',fname
   f = h5py.File(fname,'r')
   m,n = f['shape'].value
   pepo = np.empty((m,n),dtype=np.object)
   for i in range(m):
      for j in range(n):
         pepo[i,j] = f['site_'+str(i)+'_'+str(j)].value 
   f.close()
   return pepo

def eval_heish(pepsa, pepsb, auxbond=None):
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
   for k in range(len(indx)):
      fname = dirname+'/spepo_'+str(k)+'.h5'
      spepo = loadPEPO(fname,iprt=1)
      val += clst[k]*evalContraction(spepo,pepsa,pepsb,auxbond)
   return val

def evalContraction(spepo,pepsa,pepsb,auxbond):
    epeps0 = np.empty(spepo.shape, dtype=np.object)
    L = pepsa.shape[0]
    Ltot = L+nf*(L-1)
    assert Ltot+2 == epeps0.shape[0]
    psites = [1+i*(nf+1) for i in range(L)]
    # Unaffected boundary parts
    for i in range(Ltot+2):
       epeps0[0,i] = spepo[0,i]
       epeps0[i,0] = spepo[i,0]
       epeps0[Ltot+1,i] = spepo[Ltot+1,i]
       epeps0[i,Ltot+1] = spepo[i,Ltot+1]
    # Unaffected interior parts
    for ii in range(1,Ltot+1):
       if ii in psites: continue
       for jj in range(1,Ltot+1):
          if jj in psites: continue
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
    for ii in range(Ltot+2):
       for jj in range(Ltot+2):
	  print ii,jj,epeps0[ii,jj].shape
    val = peps.contract_cpeps(epeps0, auxbond)
    print val
    exit()
    return val
