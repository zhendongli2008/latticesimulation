import h5py
import numpy

flter = 'lzf'

def saveMPS(fname,mps,iprt=0):
   if iprt>0: print '[ioPEPO.saveMPS] fname=',fname
   f = h5py.File(fname,'w')
   n = len(mps)
   f['shape'] = [n]
   for i in range(n):
      f.create_dataset('site_'+str(i), data=mps[i], compression=flter)
   f.close()
   return 0

def loadMPS(fname,iprt=0):
   if iprt>0: print '[ioPEPO.loadMPS] fname=',fname
   f = h5py.File(fname,'r')
   n = f['shape'][0]
   mps = numpy.empty(n,dtype=numpy.object)
   for i in range(n):
      mps[i] = f['site_'+str(i)].value
   f.close()
   return mps

def savePEPO(fname,pepo,iprt=0):
   if iprt>0: print '[ioPEPO.savePEPO] fname=',fname
   f = h5py.File(fname,'w')
   m,n = pepo.shape
   f['shape'] = [m,n]
   for i in range(m):
      for j in range(n):
         f.create_dataset('site_'+str(i)+'_'+str(j), data=pepo[i,j], compression=flter)
   f.close()
   return 0

def loadPEPO(fname,iprt=0):
   if iprt>0: print '[ioPEPO.loadPEPO] fname=',fname
   f = h5py.File(fname,'r')
   m,n = f['shape'].value
   pepo = numpy.empty((m,n),dtype=numpy.object)
   for i in range(m):
      for j in range(n):
         pepo[i,j] = f['site_'+str(i)+'_'+str(j)].value 
   f.close()
   return pepo
