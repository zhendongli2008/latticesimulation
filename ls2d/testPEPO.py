from isingMapping import mass2c
import h5py
import genFit
import genPEPO
import ioPEPO
import genPEPOsmall
import numpy
import contraction2d

dirname = 'tmp2'
ng = 2
n = 101
center = (n/2,n/2)
mass2lst = genFit.genMass2lst(mass2c,50,28)

info = [dirname,ng,n,center,mass2lst]
#genFit.checkData(info,iop=0)
ifdump = False #True
if ifdump: 
   indx,mlst,clst = genFit.fitCoulomb(info,k=10,nselect=15,ifplot=False)

f = h5py.File(dirname+'/fitCoulomb.h5','r')
indx = f['indx_final'].value 
mlst = f['mlst_final'].value  
clst = f['clst_final'].value  
f.close()
print 'indx=',indx
print 'mlst=',mlst
print 'clst=',clst

L = 4
nf = 0
abond = 20
ic = 3
k = 0
if ifdump:
   coeff = clst[k]
   mass2 = mlst[k]
   print 'k=',k,'coeff=',coeff
   npepo = genPEPO.genNPEPO(n,mass2,ng,iprt=1,auxbond=abond,iop=1,\
        	 	    nij=None)
   fname = dirname+'/pepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
   ioPEPO.savePEPO(fname,npepo,iprt=1)
   spepo = genPEPOsmall.genBPEPO(npepo,L,nf,auxbond=abond)
   fname = dirname+'/spepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
   ioPEPO.savePEPO(fname,spepo,iprt=1)
else:
   fname = dirname+'/pepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
   npepo = ioPEPO.loadPEPO(fname,iprt=1)
   fname = dirname+'/spepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
   spepo = ioPEPO.loadPEPO(fname,iprt=1)
   
   epeps = numpy.empty(spepo.shape,dtype=numpy.object)
   nn = spepo.shape[0]
   for i in range(nn):
      epeps[i,0] = spepo[i,0]
      epeps[0,i] = spepo[0,i]
      epeps[nn-1,i] = spepo[nn-1,i]
      epeps[i,nn-1] = spepo[i,nn-1]
   for i in range(1,nn-1):
      for j in range(1,nn-1):
         epeps[i,j] = spepo[i,j][0,0]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print 'cab=',cab
   
   epeps = numpy.empty(npepo.shape,dtype=numpy.object)
   nn = npepo.shape[0]
   for i in range(nn):
      for j in range(nn):
         epeps[i,j] = npepo[i,j][0,0]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print 'cab=',cab
 
