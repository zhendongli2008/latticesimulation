import numpy
import mps
import contraction2d
import copy

def genBPEPO(pepo,L,nf,auxbond=20):
   print '[genPEPOsmall.genBPEPO] shape=',pepo.shape,'(L,nf)=',(L,nf),\
	 'auxbond=',auxbond
   ntot = pepo.shape[0]
   Ltot = L + nf*(L-1)
   dist = nf+1
   nl = (ntot-Ltot)/2 
   nr = ntot-nl-Ltot # >= nl
   print ' Ltot=',Ltot,' dist=',dist,' nl=',nl,' nr=',nr,' ratio=',float(ntot)/Ltot
   # Bottom MPS
   bmps = numpy.empty(ntot,dtype=numpy.object)
   for j in range(ntot):
      l,u,d,r = pepo[0,j][0,0].shape
      assert d == 1
      bmps[j] = pepo[0,j][0,0].reshape(l,u*d,r)
   # Compression 
   for i in range(1,nl):
      cmpo = numpy.empty(ntot,dtype=numpy.object)
      for j in range(ntot):
         cmpo[j] = pepo[i,j][0,0].copy()
      bmps = contraction2d.mpo_mapply(cmpo,bmps)
      if auxbond is not None: # compress
         bmps = mps.compress(bmps,auxbond)
   # Upper MPS -> Note that the MPO is not upper-lower symmetric !
   umps = numpy.empty(ntot,dtype=numpy.object)
   for j in range(ntot):
      l,u,d,r = pepo[ntot-1,j][0,0].shape
      assert u == 1
      umps[j] = pepo[ntot-1,j][0,0].reshape(l,u*d,r)
   # Compression 
   for i in range(ntot-2,ntot-nr-1,-1):
      cmpo = numpy.empty(ntot,dtype=numpy.object)
      for j in range(ntot):
 	 cmpo[j] = pepo[i,j][0,0].transpose(0,2,1,3).copy() # ludr->ldur
      umps = contraction2d.mpo_mapply(cmpo,umps)
      if auxbond is not None: # compress
         umps = mps.compress(umps,auxbond)
   #
   # TPEPO
   #
   tpepo = numpy.empty((Ltot+2,ntot),dtype=numpy.object)
   for j in range(ntot):
      l,u,r = bmps[j].shape
      tpepo[0,j] = bmps[j].reshape((1,1,l,u,1,r)).copy() # Add physical index
   for j in range(ntot):
      l,d,r = umps[j].shape
      tpepo[Ltot+1,j] = umps[j].reshape((1,1,l,1,d,r)).copy() # Add physical index
   for i in range(Ltot):
      for j in range(ntot):
         tpepo[i+1,j] = pepo[i+nl,j].copy()
 
   epeps = numpy.empty(pepo.shape,dtype=numpy.object)
   for i in range(pepo.shape[0]):
      for j in range(pepo.shape[1]):
         epeps[i,j] = pepo[i,j][0,0]
   epeps[3,3] = pepo[3,3][1,1]
   epeps[4,4] = pepo[4,4][1,1]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print 'TEST2-cab=',cab
  
   epeps = numpy.empty(tpepo.shape,dtype=numpy.object)
   for i in range(tpepo.shape[0]):
      for j in range(tpepo.shape[1]):
         epeps[i,j] = tpepo[i,j][0,0]
   epeps[3,3] = tpepo[3,3][1,1]
   epeps[4,4] = tpepo[4,4][1,1]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print 'TEST2-cab=',cab
   epeps = numpy.empty(tpepo.shape,dtype=numpy.object)
   for i in range(tpepo.shape[0]):
      for j in range(tpepo.shape[1]):
         epeps[i,j] = tpepo[i,j][0,0]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print 'TEST1-cab=',cab
   exit()
 
   # Left MPS
   lmps = [None]*(Ltot+2)
   for i in range(Ltot+2):
      l,u,d,r = tpepo[i,0][0,0].shape
      assert l == 1
      lmps[i] = numpy.reshape(tpepo[i,0][0,0], (u,d,r)).transpose(1,2,0) # udr->dru
   for i in range(1,nl):
      cmpo = [None]*(Ltot+2)
      for j in range(Ltot+2):
         cmpo[j] = tpepo[j,i][0,0].transpose(2,3,0,1) # ludr->drlu
      lmps = contraction2d.mpo_mapply(cmpo,lmps)
      if auxbond is not None: # compress
         lmps = mps.compress(lmps,auxbond)
   # Right MPS
   rmps = [None]*(Ltot+2)
   for i in range(Ltot+2):
      l,u,d,r = tpepo[i,ntot-1][0,0].shape
      assert r == 1
      rmps[i] = numpy.reshape(tpepo[i,ntot-1][0,0], (l,u,d)).transpose(2,0,1) # lud->dlu
   for i in range(1,nr):
      cmpo = [None]*(Ltot+2)
      for j in range(Ltot+2):
         cmpo[j] = tpepo[j,ntot-i-1][0,0].transpose(2,0,3,1) # ludr->dlru
      rmps = contraction2d.mpo_mapply(cmpo,rmps)
      if auxbond is not None: # compress
         rmps = mps.compress(rmps,auxbond)
   #
   # Assemble SPEPO
   #
   spepo = numpy.empty((Ltot+2,Ltot+2),dtype=numpy.object)
   # Middle
   for i in range(Ltot):
      for j in range(Ltot):
	 spepo[i+1,j+1] = pepo[i+nl,j+nl].copy()
   # Left 
   for j in range(Ltot+2):
      d,r,u = lmps[j].shape
      spepo[j,0] = numpy.reshape(lmps[j],(1,d,r,u)).transpose(0,3,1,2) # ldru->ludr
   # Right
   for j in range(Ltot+2):
      d,l,u = rmps[j].shape
      spepo[j,Ltot+1] = numpy.reshape(rmps[j],(d,l,u,1)).transpose(1,2,0,3) # dlur->ludr
   # Bottom
   for j in range(Ltot):
      spepo[0,j+1] = tpepo[0,j+nl][0,0].copy() 
   # Up
   for j in range(Ltot):
      spepo[Ltot+1,j+1] = tpepo[Ltot+1,j+nl][0,0].copy()
   return spepo


if __name__ == '__main__':
   from isingMapping import mass2c
   import contraction2d
   import genPEPO 
   ng = 2
   n = 7 
   center = (n/2,n/2)
   mass2 = mass2c
   L = 4
   nf = 0
   abond = 20
   
   # NPEPO 
   npepo = genPEPO.genNPEPO(n,mass2,ng,iprt=1,auxbond=abond,iop=1,nij=None)
   # Test-1
   epeps = numpy.empty(npepo.shape,dtype=numpy.object)
   nn = npepo.shape[0]
   for i in range(nn):
      for j in range(nn):
         epeps[i,j] = npepo[i,j][0,0]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print
   print 'TEST1-cab=',cab
   # Test-2
   epeps = numpy.empty(npepo.shape,dtype=numpy.object)
   nn = npepo.shape[0]
   for i in range(nn):
      for j in range(nn):
         epeps[i,j] = npepo[i,j][0,0]
   epeps[3,3] = npepo[3,3][1,1]
   epeps[4,4] = npepo[4,4][1,1]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print 'TEST2-cab=',cab,'benchmark=',\
         genPEPO.pepo2cpeps(npepo,[(3,3)],[(4,4)],auxbond=abond)[0,0]

   # SPEPO 
   spepo = genBPEPO(npepo,L,nf,auxbond=abond)
   # TEST-1
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
   print
   print 'TEST1-cab=',cab
   # TEST-1
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
   epeps[3,3] = spepo[3,3][1,1]
   epeps[4,4] = spepo[4,4][1,1]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print 'TEST2-cab=',cab
