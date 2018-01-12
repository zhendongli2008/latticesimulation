import numpy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt

def genT1d(n):
   bc = 'OBC'
   tij = numpy.diag([2.0]*n)
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -1.0
   if bc == 'PBC':
      tij[0,n-1] = tij[n-1,0] = -1.0
   elif bc == 'OBC':
      pass
   else:
      tij[0,0] = 0.
      tij[n-1,n-1] = 0.
   return tij

def genT2d(n,mass=0.0):
   tij = genT1d(n)
   iden = numpy.identity(n)
   mat = numpy.einsum('ij,ab->iajb',tij,iden)\
       + numpy.einsum('ij,ab->iajb',iden,tij)
   mat = mat + mass**2*numpy.einsum('ij,ab->iajb',iden,iden)
   return mat 

def test_1d(m):
   n = 2*m+1
   t1d = genT1d(n)
   t1d = t1d + 1.e-3*numpy.identity(n)
   tinv = scipy.linalg.inv(t1d)
   plt.plot(tinv[m],'ro-')
   plt.plot(tinv[m-1],'go-')
   plt.plot(tinv[m+1],'go-')
   plt.plot(tinv[m-5],'bo-')
   plt.plot(tinv[m+5],'bo-')
   tm = numpy.max(tinv)
   plt.ylim([0.0,tm*1.01])
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

def genPairs(tinv,n,i,j):
   dist = distance(n,i,j)
   nmm = neighbor(n,i,j)
   dxy = map(lambda x:dist[x],nmm)
   vxy = map(lambda x:tinv[i,j][x],nmm)
   return dxy,vxy

#  (m,m+k)-(m+k,m+k)
#     |        |
#   (m,m)---(m+k,m)
def matshow(tinv,m,k=5):
   plt.matshow(tinv[m,m])
   plt.show()
   plt.matshow(tinv[m+k,m])
   plt.show()
   plt.matshow(tinv[m,m+k])
   plt.show()
   plt.matshow(tinv[m+k,m+k])
   plt.show()
   return 0

def curveplot(mass=0.01,k=2,m=30):
   n = 2*m+1
   print 'm,n=',(m,n),'mass=',mass

   def coord(x,y):
      return '('+str(x)+','+str(y)+')'
   def f1(x,a,b,c):
      return a*numpy.exp(-b*x)+c
   def f2(x,a,b,c):
      return a*scipy.special.k0(b*x)+c
   def fit(d0,v0):
      diameter = m/2.0
      print '\nfit diameter=',diameter
      c = map(lambda x:x[0],numpy.argwhere(numpy.array(d0)<=diameter))
      d0r = numpy.array([d0[i] for i in c])
      v0r = numpy.array([v0[i] for i in c])
      order = numpy.argsort(d0r)
      d0r = d0r[order]
      v0r = v0r[order]
      popt1 = scipy.optimize.curve_fit(f1,d0r,v0r)[0]
      errs1 = numpy.abs(f1(d0r,*popt1)-v0r)
      print ' errs1-exp',numpy.max(errs1)
      print ' popt1-exp',popt1
      popt2 = scipy.optimize.curve_fit(f2,d0r,v0r,\
		      		       p0=[1./(2.0*numpy.pi),mass,0.])[0]
      errs2 = numpy.abs(f2(d0r,*popt2)-v0r)
      print ' errs2-bes',numpy.max(errs2)
      print ' popt2-bes',popt2
      return popt1,popt2,errs1,errs2,d0r,v0r

   t2d = genT2d(n,mass)
   t2d = t2d.reshape((n*n,n*n))
   tinv = scipy.linalg.inv(t2d)
   tinv = tinv.reshape((n,n,n,n))

   d0,v0 = genPairs(tinv,n,m,m)
   d1,v1 = genPairs(tinv,n,m+k,m)
   d2,v2 = genPairs(tinv,n,m+k,m+k)
   
   plt.plot(d0,v0,'bo',label='(m,m)='+coord(m,m)+' n='+str(n))
   plt.plot(d1,v1,'g+',label='(m+k,m)='+coord(m+k,m)+' n='+str(n))
   plt.plot(d2,v2,'rx',label='(m+k,m+k)='+coord(m+k,m+k)+' n='+str(n))
  
   iffit = True
   if iffit:
      x = numpy.linspace(1.e-8,50,1000)
      popt1a,popt2a,e1a,e2a,d0r,v0r = fit(d0,v0)
      popt1b,popt2b,e1b,e2b,d1r,v1r = fit(d1,v1)
      popt1c,popt2c,e1c,e2c,d2r,v2r = fit(d2,v2)
      #plt.plot(x,f1(x,*popt1a),'k-',label='fitted-exp')
      #plt.plot(x,f1(x,*popt1b),'k-',label='fitted-exp')
      #plt.plot(x,f1(x,*popt1c),'k-',label='fitted-exp')
      plt.plot(x,f2(x,*popt2a),'b--',label='fitted-bes')
      plt.plot(x,f2(x,*popt2b),'g--',label='fitted-bes')
      plt.plot(x,f2(x,*popt2c),'r--',label='fitted-bes')

   plt.xlim([0,4*k])
   tm = numpy.max(tinv)
   plt.ylim([-0.1,tm*1.01])
   plt.legend()
   plt.savefig('data/fr.pdf')
   plt.show()

   iferr= True
   if iffit and iferr:
      # Error 
      #plt.semilogy(d0r,e1a,'bo-',label='c-exp')
      #plt.semilogy(d1r,e1b,'bo-',label='v-exp')
      #plt.semilogy(d2r,e1c,'go-',label='d-exp')
      plt.semilogy(d0r,e2a,'ro-',label='c-bes')
      plt.semilogy(d1r,e2b,'bo-',label='v-bes')
      plt.semilogy(d2r,e2c,'go-',label='d-bes')
      plt.xlim([0,numpy.max(d0r)*1.2])
      plt.legend()
      plt.savefig('data/err.pdf')
      plt.show()
   return 0

def fitCoulomb(k=5,m=10):
   nk = 2*k+1
   n = 2*m+1
   mlst = [0.004096,0.01024,0.0256,0.064,0.16,0.4,1.,2.5,6.25,15.625,39.0625,97.6563,244.141]
   terms = len(mlst)
   coeff = numpy.zeros((terms,n,n))
   tc = numpy.zeros((terms,n,n))
   tv = numpy.zeros((terms,n,n))
   td = numpy.zeros((terms,n,n))
   for i,mass in enumerate(mlst):
      t2d = genT2d(n,mass)
      t2d = t2d.reshape((n*n,n*n))
      tinv = scipy.linalg.inv(t2d)
      tinv = tinv.reshape((n,n,n,n))
      coeff[i] = tinv[m,m]
      # save
      tc[i] = tinv[m,m]
      tv[i] = tinv[m,m+k]
      td[i] = tinv[m+k,m+k]
   coeff = coeff.reshape(terms,n,n)
   coeff = coeff.transpose(1,2,0) # (n,n,terms)
   # Only consider the central point
   dist = distance(n,m,m)
   # Fitting can use a much larger region than physical region
   fmm = neighbor(n,m,m)
   dxy = numpy.array(map(lambda x:dist[x],fmm))
   fit_radial = m/2.0
   c = map(lambda x:x[0],numpy.argwhere(dxy<=fit_radial))
   dxy = numpy.array([dxy[i] for i in c])
   fmm = numpy.array([fmm[i] for i in c])
   coeff = numpy.vstack(map(lambda x:coeff[x[0],x[1]],fmm))
   # RHS
   vxy = 1.0/dxy
   wt = numpy.ones(vxy.shape)
   bvec = coeff.T.dot(wt*vxy)
   amat = coeff.T.dot(numpy.einsum('i,ij->ij',wt,coeff))
   clst = numpy.linalg.solve(amat,bvec)
   plt.plot(clst/abs(clst)*numpy.log10(abs(clst)),'ro-')
   plt.savefig('data/clst.pdf')
   plt.show()
   print 'clst=',clst
   tc2 = numpy.zeros((n,n))
   tv2 = numpy.zeros((n,n))
   td2 = numpy.zeros((n,n))
   for i,mass in enumerate(mlst):
      tc2 += tc[i]*clst[i]
      tv2 += tv[i]*clst[i]
      td2 += td[i]*clst[i]

   def genPairs(rdist,tinv,n,i,j):
      dist = distance(n,i,j)
      nmm = neighbor(n,i,j)
      dxy = numpy.array(map(lambda x:dist[x],nmm))
      c = map(lambda x:x[0],numpy.argwhere(dxy<=rdist))
      dxy = numpy.array([dxy[k] for k in c])
      nmm = numpy.array([nmm[k] for k in c])
      vxy = numpy.array(map(lambda x:tinv[x[0],x[1]],nmm))
      order = numpy.argsort(dxy)
      dxy = dxy[order]
      vxy = vxy[order] 
      return dxy,vxy

   # Measurement
   diameter = 2*k*numpy.sqrt(2)
   d0,v0 = genPairs(diameter,tc2,n,m,m)
   d1,v1 = genPairs(diameter,tv2,n,m,m+k)
   d2,v2 = genPairs(diameter,td2,n,m+k,m+k)
   # Error
   print 'Summary of fitting:'
   print 'nk=',nk
   print 'n=',n
   print 'terms=',terms
   print 'npoint=',len(fmm)
   print 'fit_radial=',fit_radial # around the center
   print 'diameter=',diameter
   print '|err|max[c,v,d]=',numpy.max(abs(v0-1.0/d0)),\
 			    numpy.max(abs(v1-1.0/d1)),\
			    numpy.max(abs(v2-1.0/d2))
   print '|err|nrm[c,v,d]=',numpy.linalg.norm(v0-1.0/d0),\
			    numpy.linalg.norm(v1-1.0/d1),\
			    numpy.linalg.norm(v2-1.0/d2)
   x = numpy.linspace(1.e-8,50,1000)
   plt.plot(x,1/x,'k-',label='1/r')
   #plt.plot(dxy,vxy,'ko',label='1/rij',markersize=8)
   plt.plot(d0,v0,'ro-',label='Vc (n='+str(n)+')',markersize=8)
   plt.plot(d1,v1,'g+-',label='Vv (n='+str(n)+')',markersize=8)
   plt.plot(d2,v2,'bx-',label='Vd (n='+str(n)+')',markersize=8)

   plt.xlim([0,1.2*diameter])
   plt.ylim([-0.1,1.5])
   plt.legend()
   plt.savefig('data/fitCoulomb_terms_'+str(terms)+'_nk_'+str(nk)+'_n_'+str(n)+'.pdf')
   plt.show()
   return 0


if __name__ == '__main__':

#===========
# Benchmark
#===========
# 0. Compute f(r) w.r.t. center - whether it is isotropic
# 1. compare different lattice size - shift measure center (i,j) to boundary 
# 2. compare different lambda?
   
   #fitCoulomb(k=10,m=30)
   curveplot(mass=0.001,k=5,m=50)
   #curveplot(mass=0.001,k=5,m=40)

