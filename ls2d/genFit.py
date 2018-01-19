# 
# Generation of data and fitting
# 
import numpy
import num2d
import h5py
import time
import matplotlib.pyplot as plt

def getFname(info):
   ng,n,center,masst = info
   fname = 'data/cij_ng'+str(ng)+'_n'+str(n)+'_center'+str(center)+'_masst'+str(masst)+'.h5'
   return fname

def saveData(info,cij):
   ng,n,center,masst = info
   fname = getFname(info)
   print '[saveData] fname=',fname 
   f = h5py.File(fname,'w')
   info = f.create_dataset("info",(1,),dtype='i')
   info.attrs['ng'] = ng
   info.attrs['n'] = n
   info.attrs['center'] = center
   info.attrs['masst'] = masst
   f['cij'] = cij
   f.close()
   return 0

def loadData(fname):
   print '[loadData] fname=',fname 
   f = h5py.File(fname,'r')
   ng = f['info'].attrs['ng'] 
   n = f['info'].attrs['n'] 
   center = f['info'].attrs['center'] 
   center = tuple(center)
   masst = f['info'].attrs['masst'] 
   cij = f['cij'].value
   info = [ng,n,center,masst]
   f.close()
   return info,cij

def genData(info):
   ng,n,center,masst = info
   print '\n[genData] ng,n,center,masst=',(ng,n,center,masst)
   palst = [center]
   pblst = [(i,j) for i in range(1,n-1) for j in range(1,n-1)]
   t0 = time.time()
   cij = num2d.correlationFunctions(n,mass2=masst,ng=ng,\
		   		    palst=palst,pblst=pblst,\
				    iprt=1)
   t1 = time.time()
   print ' total time = ',t1-t0
   cij = cij[0].reshape(n-2,n-2)
   saveData(info,cij)
   return 0

def genFit():
   mlst = numpy.array([0.004096,0.01024,0.0256,0.064,0.16,0.4,1.,2.5,6.25,15.625,39.0625,97.6563,244.141])
   n = 4

if __name__ == '__main__':
   from isingMapping import mass2c
   info = [2,11,(5,5),mass2c]
   genData(info)
   fname = getFname(info)
   info,cij = loadData(fname)
   cij = cij*(4.0+mass2c)
   plt.matshow(cij)
   plt.show()
   plt.plot(cij[4],'ro-')
   plt.show()
   print cij
#
#  total time =  5.20750689507
# [saveData] fname= data/cij_ng2_n11_center(5, 5)_masst-1.73081468579.h5
# [loadData] fname= data/cij_ng2_n11_center(5, 5)_masst-1.73081468579.h5
# [[ 0.20622927  0.25064401  0.28667343  0.31193584  0.3215822   0.31193584
#    0.28667343  0.25064401  0.20622927]
#  [ 0.25064401  0.30433939  0.35096454  0.38708497  0.40288919  0.38708497
#    0.35096454  0.30433939  0.25064401]
#  [ 0.28667343  0.35096454  0.41258014  0.46886453  0.50159129  0.46886453
#    0.41258014  0.35096454  0.28667343]
#  [ 0.31193584  0.38708497  0.46886453  0.56423766  0.65630225  0.56423766
#    0.46886453  0.38708497  0.31193584]
#  [ 0.3215822   0.40288919  0.50159129  0.65630225  1.          0.65630225
#    0.50159129  0.40288919  0.3215822 ]
#  [ 0.31193584  0.38708497  0.46886453  0.56423766  0.65630225  0.56423766
#    0.46886453  0.38708497  0.31193584]
#  [ 0.28667343  0.35096454  0.41258014  0.46886453  0.50159129  0.46886453
#    0.41258014  0.35096454  0.28667343]
#  [ 0.25064401  0.30433939  0.35096454  0.38708497  0.40288919  0.38708497
#    0.35096454  0.30433939  0.25064401]
#  [ 0.20622927  0.25064401  0.28667343  0.31193584  0.3215822   0.31193584
#    0.28667343  0.25064401  0.20622927]]
# 
