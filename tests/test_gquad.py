import numpy
import scipy.linalg
import matplotlib.pyplot as plt

def fun(gamma=1.0):
   return 2*numpy.pi/numpy.sqrt(4.0-gamma**2)

def num(gamma=1.0,ng=10):
   xts,wts = numpy.polynomial.hermite.hermgauss(ng)
   print 'xts=',xts
   print 'wts=',wts
   wij = numpy.exp(gamma*numpy.einsum('i,j->ij',xts,xts))
   return numpy.einsum('i,ij,j',wts,wij,wts)
 
def test():
   x = numpy.linspace(-1.9,1.9,20)
   y = [fun(i) for i in x]
   plt.plot(x,y,'ro-',label='exact')
   y1 = [num(i,ng=5) for i in x]
   plt.plot(x,y1,'go-',label='num10')
   plt.legend()
   plt.show()
   return 0

def genArep(ng):
   a = numpy.zeros((ng,ng))
   for i in range(ng-1):
      a[i,i+1] = numpy.sqrt(i+1)
   return a

def genXrep(ng):
   a = genArep(ng)
   x = 1.0/numpy.sqrt(2.0)*(a+a.T) 
   return x

def alg(gamma=1.0,ng=10):
   x = genXrep(ng)
   e = gamma*numpy.kron(x,x)
   e = scipy.linalg.expm(e)[0,0]*numpy.pi # N^4 due to two oscillators
   return e

if __name__ == '__main__':
   #test()
   #print genXrep(10)
   #print scipy.linalg.eigh(genXrep(10))[0]
   print num(1.9,ng=10),fun(1.9)
   print alg(1.9,ng=10),fun(1.9)
   print num(-1.3,ng=3),fun(-1.3)
   print alg(-1.3,ng=3),fun(-1.3)

   # Exponential convergence
   ngMax = 100
   for gamma in [1.0,1.5,1.98]: 
      grd = range(1,ngMax)
      err1 = numpy.log10([abs(num(gamma,ng=i)-fun(gamma)) for i in range(1,ngMax)])
      #err2 = numpy.log10([abs(alg(gamma,ng=i)-fun(gamma)) for i in range(1,ngMax)])
      fac = numpy.log10(gamma/2.0)
      err3 = [i*fac+numpy.log10(numpy.pi/(2.0-gamma)) for i in range(1,ngMax)]
      plt.plot(grd,err1,'ro-')
      #plt.plot(grd,err2,'b+-')
      plt.plot(grd,err3,'go-')
   plt.show()
