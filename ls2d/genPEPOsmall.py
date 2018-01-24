import numpy
import mps
import contraction2d
import copy

def genPSites(ntot,Lphys,nf):
   Ltot = Lphys + nf*(Lphys-1)
   nl = (ntot-Ltot)/2 
   nr = ntot-nl-Ltot # >= nl
   psites = [nl+i*(nf+1) for i in range(Lphys)]
   return psites

def genBPEPO(pepo,Lphys,nf,auxbond=20):
   auxbond_hor = int(1.5*auxbond)
   print '\n[genPEPOsmall.genBPEPO] auxbond=',auxbond,' auxbond_hor=',auxbond_hor
   ntot = pepo.shape[0]
   Ltot = Lphys + nf*(Lphys-1) 
   dist = nf+1
   nl = (ntot-Ltot)/2 
   nr = ntot-nl-Ltot # >= nl
   print ' ntot=',pepo.shape[0],'(Lphys,nf)=',(Lphys,nf)
   print ' Ltot=',Ltot,' dist=',dist,' nl=',nl,' nr=',nr,' ratio=',float(ntot)/Ltot
   # Bottom MPS
   bmps = [None]*ntot
   for j in range(ntot):
      l,u,d,r = pepo[0,j][0,0].shape
      assert d == 1
      bmps[j] = pepo[0,j][0,0].reshape(l,u*d,r)
   # Compression 
   for i in range(1,nl):
      cmpo = [None]*ntot
      for j in range(ntot):
	 cmpo[j] = pepo[i,j][0,0].copy()
      bmps = contraction2d.mpo_mapply(cmpo,bmps)
      if auxbond is not None: # compress
         bmps = mps.compress(bmps,auxbond)
   # Upper MPS -> Note that the MPO is not upper-lower symmetric !
   umps = [None]*ntot
   for j in range(ntot):
      l,u,d,r = pepo[ntot-1,j][0,0].shape
      assert u == 1
      umps[j] = pepo[ntot-1,j][0,0].reshape(l,u*d,r)
   # Compression 
   for i in range(ntot-2,ntot-nr-1,-1):
      cmpo = [None]*ntot
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
      tpepo[0,j] = bmps[j].reshape((1,1,l,u,1,r)) # Add physical index
   for j in range(ntot):
      l,d,r = umps[j].shape
      tpepo[Ltot+1,j] = umps[j].reshape((1,1,l,1,d,r)) # Add physical index
   for i in range(Ltot):
      for j in range(ntot):
         tpepo[i+1,j] = pepo[i+nl,j].copy()
   # Left MPS
   lmps = [None]*(Ltot+2)
   for i in range(Ltot+2):
      l,u,d,r = tpepo[i,0][0,0].shape
      assert l == 1
      lmps[i] = numpy.reshape(tpepo[i,0][0,0], (u,d,r)).transpose(1,2,0) # udr->dru
   for j in range(1,nl):
      cmpo = [None]*(Ltot+2)
      for i in range(Ltot+2):
         cmpo[i] = tpepo[i,j][0,0].transpose(2,3,0,1) # ludr->drlu
      lmps = contraction2d.mpo_mapply(cmpo,lmps)
      if auxbond is not None: # compress
         lmps = mps.compress(lmps,auxbond_hor)
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
         rmps = mps.compress(rmps,auxbond_hor)
   #
   # Assemble SPEPO
   #
   spepo = numpy.empty((Ltot+2,Ltot+2),dtype=numpy.object)
   # Middle
   for i in range(Ltot):
      for j in range(Ltot):
	 spepo[i+1,j+1] = tpepo[i+1,j+nl].copy()
   # Left 
   for i in range(Ltot+2):
      d,r,u = lmps[i].shape
      l = 1
      tmp = numpy.reshape(lmps[i],(l,d,r,u)).transpose(0,3,1,2) # ldru->ludr
      spepo[i,0] = tmp.reshape((1,1,l,u,d,r))
   # Right
   for i in range(Ltot+2):
      d,l,u = rmps[i].shape
      r = 1
      tmp = numpy.reshape(rmps[i],(d,l,u,r)).transpose(1,2,0,3) # dlur->ludr
      spepo[i,Ltot+1] = tmp.reshape((1,1,l,u,d,r))
   # Bottom
   for j in range(Ltot):
      spepo[0,j+1] = tpepo[0,j+nl].copy() 
   # Up
   for j in range(Ltot):
      spepo[Ltot+1,j+1] = tpepo[Ltot+1,j+nl].copy()
   return spepo


if __name__ == '__main__':
   from isingMapping import mass2c
   import contraction2d
   import genPEPO 
   ng = 2
   n = 31 
   center = (n/2,n/2)
   mass2 = mass2c
   Lphys = 4
   nf = 0
   abond = 20 # Note that npepo seems to be more accurate !!!
   Ltot = Lphys + nf*(Lphys-1) 
  
   pa0 = (1,1) #Ltot/2,Ltot/2) # on spepo
   pb0 = (2,2) #Ltot/2+1,Ltot/2+1)
   def address(pos):
      nl = (n-Ltot)/2 
      ii = pos[0] + nl - 1 # -1 due to the additional boundary 
      jj = pos[1] + nl - 1
      return (ii,jj) 

   # NPEPO 
   npepo = genPEPO.genNPEPO(n,mass2,ng,iprt=1,auxbond=abond,iop=1,nij=None)
   # SPEPO 
   spepo = genBPEPO(npepo,Lphys,nf,auxbond=abond)
   
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
   pa = address(pa0)
   pb = address(pb0)
   epeps[pa] = npepo[pa][1,1]
   epeps[pb] = npepo[pb][1,1]
   cab = contraction2d.contract(epeps,auxbond=abond)
   print 'TEST2(npepo)-cab=',cab,'benchmark=',\
         genPEPO.pepo2cpeps(npepo,[pa],[pb],auxbond=abond)[0,0]
