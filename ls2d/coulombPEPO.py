import h5py
import numpy
import genFit
import genPEPO
import ioPEPO
import genPEPOsmall
import contraction2d
from isingMapping import mass2c
import time

dirname = 'tmp2'
ng = 2
n = 101
center = (n/2,n/2)
mass2lst = genFit.genMass2lst(mass2c,50,28)

info = [dirname,ng,n,center,mass2lst]

ifsave = False #True
if ifsave: 
   indx,mlst,clst = genFit.fitCoulomb(info,k=10,nselect=15,ifplot=False)
else:
   f = h5py.File(dirname+'/fitCoulomb.h5','r')
   indx = f['indx_final'].value 
   mlst = f['mlst_final'].value  
   clst = f['clst_final'].value  
   f.close()
   print 'indx=',indx
   print 'mlst=',mlst
   print 'clst=',clst

import numpy as np
Sz = np.array([[1.,0.],[0.,-1.]])*0.5
Sm = np.array([[0.,1.],[0., 0.]])
Sp = Sm.T

ifdebug = True
if ifdebug:
   Pairs = [[]]
   nindx = len(indx)
else:
   Pairs = [[Sz,Sz],[Sp,Sm],[Sm,Sp]]
   nindx = len(indx)
nc = len(Pairs)

n = 51
L = 4
nf = 1
dist = nf+1.0
abond = 25
psites = genPEPOsmall.genPSites(n,L,nf)
pa0 = (1,1)
pb0 = (1,5)

if ifsave:
   for k in range(nindx):
      for ic in range(nc):
         coeff = clst[k]
         mass2 = mlst[k]
         print 'k=',k,'coeff=',coeff,'mass2=',mass2
         npepo = genPEPO.genNPEPO(n,mass2,ng,iprt=1,auxbond=abond,iop=1,\
              	 	          nij=Pairs[ic],psites=psites,fac=coeff)
         fname = dirname+'/pepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
         ioPEPO.savePEPO(fname,npepo,iprt=1)
         spepo = genPEPOsmall.genBPEPO(npepo,L,nf,auxbond=abond)
         fname = dirname+'/spepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
         ioPEPO.savePEPO(fname,spepo,iprt=1)
else:

   clst[:] = 1.0 # for the new convesion
   Ltot = L + nf*(L-1) 
   def address(pos):
      nl = (n-Ltot)/2 
      ii = pos[0] + nl - 1 # -1 due to the additional boundary 
      jj = pos[1] + nl - 1
      return (ii,jj) 
   pa = address(pa0)
   pb = address(pb0)
   print 'pa=',pa
   print 'pb=',pb
   
   ic = 0 
   k = 0
   fname = dirname+'/pepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
   npepo = ioPEPO.loadPEPO(fname,iprt=1)
   #print 'npepo.shape=',npepo.shape
   fname = dirname+'/spepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
   spepo = ioPEPO.loadPEPO(fname,iprt=1)
   #print 'spepo.shape=',spepo.shape

   # TEST-1
   epeps = numpy.empty(spepo.shape,dtype=numpy.object)
   nn = spepo.shape[0]
   for i in range(nn):
      for j in range(nn):
         epeps[i,j] = spepo[i,j][0,0]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print
   print 'TEST1-cab=',cab
   # TEST-1
   epeps = numpy.empty(spepo.shape,dtype=numpy.object)
   nn = spepo.shape[0]
   for i in range(nn):
      for j in range(nn):
         epeps[i,j] = spepo[i,j][0,0]
   epeps[pa0] = spepo[pa0][1,1]
   epeps[pb0] = spepo[pb0][1,1]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print 'TEST2-cab=',cab
   
   # Test-1
   epeps = numpy.empty(npepo.shape,dtype=numpy.object)
   nn = npepo.shape[0]
   for i in range(nn):
      for j in range(nn):
         epeps[i,j] = npepo[i,j][0,0]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print
   print 'TEST1(npepo)-cab=',cab
   # Test-2
   epeps = numpy.empty(npepo.shape,dtype=numpy.object)
   nn = npepo.shape[0]
   for i in range(nn):
      for j in range(nn):
         epeps[i,j] = npepo[i,j][0,0]
   epeps[pa] = npepo[pa][1,1]
   epeps[pb] = npepo[pb][1,1]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print 'TEST2(npepo)-cab=',cab,'benchmark=',\
         genPEPO.pepo2cpeps(npepo,[pa],[pb],auxbond=abond)[0,0]
  
   val1 = 0.
   val2 = 0. 
   for k in range(nindx):
      for ic in range(nc):
         fname = dirname+'/pepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
         npepo = ioPEPO.loadPEPO(fname,iprt=0)
         #print 'npepo.shape=',npepo.shape
         fname = dirname+'/spepo_nf'+str(nf)+'_ic'+str(ic)+'_k'+str(k)+'.h5'
         spepo = ioPEPO.loadPEPO(fname,iprt=0)
         #print 'spepo.shape=',spepo.shape

         t0 = time.time()
	 epeps = numpy.empty(spepo.shape,dtype=numpy.object)
         nn = spepo.shape[0]
         for i in range(nn):
            for j in range(nn):
               epeps[i,j] = spepo[i,j][0,0]
         epeps[pa0] = spepo[pa0][1,1]
         epeps[pb0] = spepo[pb0][1,1]
         cab1 = contraction2d.contract(epeps,auxbond=abond)
         val1 += cab1*clst[k] 
         t1 = time.time()
         print 'k=',k,clst[k],'TEST2-cab=',cab1,'v=',val1*dist,'t=',t1-t0
         # Test-2
         epeps = numpy.empty(npepo.shape,dtype=numpy.object)
         nn = npepo.shape[0]
         for i in range(nn):
            for j in range(nn):
               epeps[i,j] = npepo[i,j][0,0]
         epeps[pa] = npepo[pa][1,1]
         epeps[pb] = npepo[pb][1,1]
         cab2 = contraction2d.contract(epeps,auxbond=abond)
         val2 += cab2*clst[k] 
         t2 = time.time()
         print 'k=',k,clst[k],'TEST2-cab=',cab2,'v=',val2*dist,'t=',t2-t1
