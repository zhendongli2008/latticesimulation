import numpy
from isingMapping import mass2c
import matplotlib.pyplot as plt
import genFit

dirname = 'tmp2'
ng = 2
n = 101
center = (n/2,n/2)
mass2lst = genFit.genMass2lst(mass2c,50,28)

ifana = False
if ifana:
   info = [dirname,ng,n,center,mass2lst]
   genFit.checkData(info,iop=0)
   indx,mlst,clst = genFit.fitCoulomb(info,k=10,nselect=10,ifplot=True)
else:
   import h5py
   import genPEPO
   f = h5py.File(dirname+'/fitCoulomb.h5','r')
   indx = f['indx_final'].value 
   mlst = f['mlst_final'].value  
   clst = f['clst_final'].value  
   f.close()
   print 'indx=',indx
   print 'mlst=',mlst
   print 'clst=',clst
   abond = 20
   # Test
   for i in range(1,3):
      palst = [center]
      pblst = [(n/2,n/2+i)]
      val = 0.0
      for k in range(len(indx)):
	 coeff = clst[k]
	 mass2 = mlst[k]
         print 'k=',k,'coeff=',coeff
         npepo = genPEPO.genNPEPO(n,mass2,ng,iprt=0,auxbond=abond,iop=1,nij=None)
         val += coeff*genPEPO.pepo2cpeps(npepo,palst,pblst,auxbond=abond)[0,0]
      print 'val=',val
      exit()
