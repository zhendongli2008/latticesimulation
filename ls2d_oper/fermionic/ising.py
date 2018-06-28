import numpy
from latticesimulation.ls2d import num2d
import matplotlib.pyplot as plt

mass2c = -1.730814685786978

def isingCorrelationFuncitons(m=10):
   n = 2*m+1
   # Exact
   # ss_tc = [0.63662]
   # Approximate
   ng = 2
   mass2 = mass2c #mass2c #6 #mass2c
   T = 4.0+mass2
   beta = 1.0/T
   print '(m,n)=',(m,n),'mass2=',mass2,'beta=',beta,'T=',T
   palst = [(m,m)] 
   pblst = [(m+i,m+i) for i in range(m)]
   cij = num2d.correlationFunctions(n,mass2=mass2,ng=ng,\
		   		    palst=palst,pblst=pblst,\
				    iprt=1)
   print 'cij=',cij
   print
   cij = cij[0]/beta
   print 'cij/beta=',cij
   return n,T,cij


if __name__ == '__main__':
   
   m = 10
   n,T,cij = isingCorrelationFuncitons(m)
   
   plt.plot(range(m),cij,'ro-',label=str(n)+'-by-'+str(n))
   plt.legend()
   plt.show()
