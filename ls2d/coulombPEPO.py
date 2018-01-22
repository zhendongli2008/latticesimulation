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

ifana = False #True
if ifana:
   info = [dirname,ng,n,center,mass2lst]
   genFit.checkData(info,iop=0)
   indx,mlst,clst = genFit.fitCoulomb(info,k=10,nselect=15,ifplot=True)
else:
   f = h5py.File(dirname+'/fitCoulomb.h5','r')
   indx = f['indx_final'].value 
   mlst = f['mlst_final'].value  
   clst = f['clst_final'].value  
   f.close()
   print 'indx=',indx
   print 'mlst=',mlst
   print 'clst=',clst
   L = 4
   nf = 4
   abond = 20
   ifsavePEPO = False #True
   if ifsavePEPO:
      for k in range(len(indx)):
         coeff = clst[k]
         mass2 = mlst[k]
         print 'k=',k,'coeff=',coeff
         npepo = genPEPO.genNPEPO(n,mass2,ng,iprt=1,auxbond=abond,iop=1,\
			 	  nij=None)
         fname = dirname+'/pepo_'+str(k)+'.h5'
         ioPEPO.savePEPO(fname,npepo,iprt=1)
         spepo = genPEPOsmall.genBPEPO(npepo,L,nf,auxbond=abond)
	 fname = dirname+'/spepo_'+str(k)+'.h5'
         ioPEPO.savePEPO(fname,spepo,iprt=1)
   else:
      # Test
      for k in range(len(indx)):
         fname = dirname+'/pepo_'+str(k)+'.h5'
         npepo = ioPEPO.loadPEPO(fname,iprt=1)
	 spepo = genPEPOsmall.genBPEPO(npepo,L,nf,auxbond=abond)
	 fname = dirname+'/spepo_'+str(k)+'.h5'
         ioPEPO.savePEPO(fname,spepo,iprt=1)
         #fname = dirname+'/spepo_'+str(k)+'.h5'
	 #spepo = ioPEPO.loadPEPO(fname,iprt=1)
	 print 'k=',k,'spepo.shape=',spepo.shape
	 for i in range(spepo.shape[0]):
	    for j in range(spepo.shape[1]):
	       print 'i,j=',(i,j),spepo[i,j].shape
         exit()
