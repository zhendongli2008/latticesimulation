import numpy

def genBPEPO(pepo,L,nf,auxbond=20):
   print '[genPEPOsmall.genBPEPO] shape=',pepo.shape,'(L,nf)=',(L,nf),\
	 'auxbond=',auxbond
   ntot = pepo.shape[0]
   Ltot = L + nf*(L-1)
   dist = nf+1
   nl = (ntot-Ltot)/2
   nr = ntot-nl-Ltot # >= nl
   print ' Ltot=',Ltot,' dist=',dist,' nl=',nl,' nr=',nr,' ratio=',float(ntot)/Ltot

   return 0
