from isingMapping import mass2c
import h5py
import genFit
import genPEPO
import ioPEPO
import genPEPOsmall

dirname = 'tmp2'
ng = 2
n = 101
center = (n/2,n/2)
mass2lst = genFit.genMass2lst(mass2c,50,28)

info = [dirname,ng,n,center,mass2lst]
#genFit.checkData(info,iop=0)
indx,mlst,clst = genFit.fitCoulomb(info,k=10,nselect=15,ifplot=False)

f = h5py.File(dirname+'/fitCoulomb.h5','r')
indx = f['indx_final'].value 
mlst = f['mlst_final'].value  
clst = f['clst_final'].value  
f.close()
print 'indx=',indx
print 'mlst=',mlst
print 'clst=',clst

import numpy as np
I = np.eye(2)
Sz = .5*np.array([[1.,0.],[0.,-1.]])
Sm = np.array([[0.,1.],[0.,0.]])
Sp = Sm.T
Pairs = [[Sz,Sz],[Sp,Sm],[Sm,Sp]]

L = 4
nf = 0
abond = 20
for ic in range(3):
 for k in range(len(indx)):
   coeff = clst[k]
   mass2 = mlst[k]
   print 'k=',k,'coeff=',coeff
   npepo = genPEPO.genNPEPO(n,mass2,ng,iprt=1,auxbond=abond,iop=1,\
  		 	    nij=Pairs[ic])
   fname = dirname+'/pepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
   ioPEPO.savePEPO(fname,npepo,iprt=1)
   spepo = genPEPOsmall.genBPEPO(npepo,L,nf,auxbond=abond)
   fname = dirname+'/spepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
   ioPEPO.savePEPO(fname,spepo,iprt=1)
