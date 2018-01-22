import numpy
import mps
import contraction2d

def getLtot(L,nf):
   Ltot = L + nf*(L-1)
   return Ltot 

def genBPEPO(pepo,L,nf,auxbond=20):
   print '[genPEPOsmall.genBPEPO] shape=',pepo.shape,'(L,nf)=',(L,nf),\
	 'auxbond=',auxbond
   ntot = pepo.shape[0]
   Ltot = getLtot(L,nf)
   dist = nf+1
   nl = (ntot-Ltot)/2
   nr = ntot-nl-Ltot # >= nl
   print ' Ltot=',Ltot,' dist=',dist,' nl=',nl,' nr=',nr,' ratio=',float(ntot)/Ltot
   # Bottom MPS
   bmps = [None]*ntot
   for i in range(ntot):
      l,u,d,r = pepo[0,i][0,0].shape
      assert d == 1
      bmps[i] = numpy.reshape(pepo[0,i][0,0], (l,u*d,r))
   for i in range(1,nl):
      cmpo = [None]*ntot
      for j in range(ntot):
         cmpo[j] = pepo[i,j][0,0]
      bmps = contraction2d.mpo_mapply(cmpo,bmps)
      if auxbond is not None: # compress
         bmps = mps.compress(bmps,auxbond)
   # Upper MPS
   if nr == nl:
      umps = [None]*ntot
      for j in range(ntot):
         umps[j] = bmps[j].copy()
   elif nr == nl+1:
      cmpo = [None]*ntot
      for j in range(ntot):
         cmpo[j] = pepo[nl,j][0,0]
      umps = contraction2d.mpo_mapply(cmpo,bmps)
      if auxbond is not None: # compress
         umps = mps.compress(umps,auxbond)
   else:
      print 'error: nl,nr=',(nl,nr)
      exit()
   #
   # TPEPO
   #
   tpepo = numpy.empty((Ltot+2,ntot),dtype=numpy.object)
   for j in range(ntot):
      l,u,r = bmps[j].shape
      tpepo[0,j] = bmps[j].reshape((1,1,l,u,1,r)) # Add physical index
   for j in range(ntot):
      l,u,r = umps[j].shape
      tpepo[Ltot+1,j] = umps[j].reshape((1,1,l,1,u,r)) # Add physical index
   for i in range(Ltot):
      for j in range(ntot):
	 tpepo[i+1,j] = pepo[i+nl,j].copy()
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
