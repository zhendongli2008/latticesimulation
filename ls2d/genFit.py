# 
# Generation of data and fitting
# 
import scipy.linalg
import numpy
import exact2d
import num2d
import h5py
import time
import matplotlib.pyplot as plt
from isingMapping import mass2c

# Use geometry series for temperature
def genMass2lst(s,l,n):
   ts = 4.0+mass2c # alpha
   tl = 4.0+l      # alpha*beta**(n-1)
   ratio = numpy.power(tl/ts,1.0/(n-1.0))
   mass2lst = [ts*ratio**i-4.0 for i in range(n)]
   return mass2lst

def getFname(info):
   dirname,ng,n,center,masst = info
   fname = dirname+'/cij_ng'+str(ng)+'_n'+str(n)+'_center'+str(center)+'_masst'+str(masst)+'.h5'
   return fname

def saveData(info,cij):
   dirname,ng,n,center,masst = info
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
   dirname,ng,n,center,masst = info
   print '\n[genData] dir,ng,n,center,masst=',(dirname,ng,n,center,masst)
   assert center[0] == center[1] == n/2
   palst = [center]
   pblst = [(i,j) for i in range(1,n/2+1) for j in range(1,n/2+1)]
   t0 = time.time()
   cij = num2d.correlationFunctions(n,mass2=masst,ng=ng,\
		   		    palst=palst,pblst=pblst,\
				    iprt=1)
   cij_full = numpy.zeros((n,n))
   idx = 0 
   for i in range(1,n/2+1):
      for j in range(1,n/2+1):
	 cij_full[i,j] = cij[0][idx] # left-up 
	 cij_full[i,n-j-1] = cij[0][idx] # right-up
	 cij_full[n-i-1,j] = cij[0][idx] # left-down
	 cij_full[n-i-1,n-j-1] = cij[0][idx] # right-down
	 idx += 1 
   #plt.matshow(cij_full)
   #plt.show()
   t1 = time.time()
   print ' total time = ',t1-t0
   saveData(info,cij_full)
   return 0

# Check the data
def checkData(info,iop=0,thresh=1.0):
   dirname,ng,n,center,mlst = info
   rpos = [i-n/2 for i in range(1,n-1)]
   for i,mass in enumerate(mlst):
      fname = getFname([dirname,ng,n,center,mass])
      cij = loadData(fname)[1]
      if iop == 1: cij=cij*(4.0+mass)
      print 'i=',i,'mass=',mass,'|vmax|=',numpy.max(cij)
      if numpy.max(cij) > thresh: 
	 print 'skipped i=',i
	 continue
      plt.plot(rpos,cij[center[0]][1:n-1],'o-') 
   plt.show()
   return 0

def distance(n,i,j):
   darray = numpy.zeros((n,n))
   for a in range(n):
      for b in range(n):
         darray[a,b] = numpy.sqrt((1.0*a-i)**2+(1.0*b-j)**2)
   return darray

def neighbor(n,i,j):
   narray = []
   for a in range(n):
      for b in range(n):
	 if (a != i or b != j):
            narray.append((a,b))
   return narray

def fitCoulomb(info,k=None,nselect=10,ifplot=False,skiplst=[]):
   dirname,ng,n,center,mlst = info
   m = center[0]
   terms = len(mlst)
   coeff = numpy.zeros((terms,n,n))
   for i,mass in enumerate(mlst):
      fname = getFname([dirname,ng,n,center,mass])
      if i in skiplst: continue
      coeff[i] = loadData(fname)[1]
   coeff = coeff.reshape(terms,n,n)
   coeff = coeff.transpose(1,2,0) # (n,n,terms)
   # Only consider the central point
   dist = exact2d.distance(n,m,m)
   # Fitting can use a much larger region than physical region
   fmm = exact2d.neighbor(n,m,m)
   dxy = numpy.array(map(lambda x:dist[x],fmm))
   fit_radial = min(m-1,m/1.5)
   c = map(lambda x:x[0],numpy.argwhere(dxy<=fit_radial))
   dxy = numpy.array([dxy[i] for i in c])
   fmm = numpy.array([fmm[i] for i in c])
   
   # Start fitting
   amat = numpy.vstack(map(lambda x:coeff[x[0],x[1]],fmm))
   bvec = 1.0/dxy
   qa,ra,pa = scipy.linalg.qr(amat,pivoting=True)
   print 'amat.shape=',amat.shape
   print 'pa=',pa
   print 'Rd=',numpy.diag(ra)
   for nterm in range(terms,0,-1):
      cols = pa[:nterm]
      clst = scipy.linalg.pinv(amat[:,cols]).dot(bvec)
      errs = abs(amat[:,cols].dot(clst)-bvec)
      print 'nterm=',nterm,' err_max=',numpy.max(errs),\
		           ' err_av=',numpy.mean(abs(errs)),\
			   ' |c|_max=',numpy.max(abs(clst))
      # Save
      if nterm == 0 or nterm == nselect:
         clst_final = clst.copy()
         indx_final = pa[:nterm]
	 mlst_final = numpy.array(mlst)[indx_final]
      if nterm == nselect: break

   # Select data within a radial=rdist
   def genPairs(rdist,tinv,n,i,j):
      dist = exact2d.distance(n,i,j)
      nmm = exact2d.neighbor(n,i,j)
      dxy = numpy.array(map(lambda x:dist[x],nmm))
      c = map(lambda x:x[0],numpy.argwhere(dxy<=rdist))
      dxy = numpy.array([dxy[k] for k in c])
      nmm = numpy.array([nmm[k] for k in c])
      vxy = numpy.array(map(lambda x:tinv[x[0],x[1]],nmm))
      order = numpy.argsort(dxy)
      dxy = dxy[order]
      vxy = vxy[order] 
      return dxy,vxy

   # Measurement of fitting quality
   coeff = coeff.transpose(2,0,1)
   vci = numpy.zeros((n,n))
   for i in range(len(clst_final)):
      vci += coeff[indx_final[i]]*clst_final[i]
   if k == None: k = int(m/2.0)
   diameter = 2.0*k*numpy.sqrt(2)
   d0,v0 = genPairs(diameter,vci,n,m,m)
   
   # Error
   print '\nSummary of fitting:'
   print 'n=',n
   print 'final terms=',len(cols)
   print 'indx_final =',indx_final
   print 'mlst_final =',mlst_final
   print 'clst_final =',clst_final
   print '---details of fitting---'
   print 'terms=',terms
   print 'npoint=',len(fmm)
   print 'fit_radial=',fit_radial # around the center
   print 'check_diameter=',diameter
   print 'nk=',2*k+1
   print '|err|max=',numpy.max(abs(v0-1.0/d0))
   print '|err|nrm=',numpy.mean(abs(v0-1.0/d0))
   # Plot results
   if ifplot:
      x = numpy.linspace(1.e-8,50,1000)
      plt.plot(x,1/x,'k-',label='1/r')
      plt.plot(dxy,bvec,'ko',label='1/rij',markersize=8)
      plt.plot(d0,v0,'ro-',label='Vc (n='+str(n)+')',markersize=8)
      plt.xlim([0,1.2*diameter])
      plt.ylim([-0.1,1.5])
      plt.legend()
      plt.show()
      # Error
      plt.semilogy(d0,abs(v0-1.0/d0),'ro-',label='Vc (n='+str(n)+')',markersize=8)
      plt.xlim([0,1.2*diameter])
      plt.legend()
      plt.show()
   # Save
   f = h5py.File(dirname+'/fitCoulomb.h5','w')
   f['indx_final'] = indx_final
   f['mlst_final'] = mlst_final
   f['clst_final'] = clst_final
   f.close()
   return indx_final,mlst_final,clst_final

if __name__ == '__main__':
   info = ['data',2,11,(5,5),mass2c]
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
