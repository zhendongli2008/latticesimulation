import numpy
import scipy.linalg
import matplotlib.pyplot as plt
import gen2d
from latticesimulation.ls2d import exact2d
from latticesimulation.ls2d import contraction2d

# no. of test points
np = 5
# note that error is large for small mass, 
# which requires a very large lattice size!
mass = 0.04
a = 1.0

def exactFun(x):
   return scipy.special.k0(mass*x)/(2*numpy.pi)
x = numpy.linspace(0.01,50.0,1000)
y = exactFun(x)
plt.plot(x,y,'k-',label='exact')

# Exact inverse
for st,m in zip(['o','x','+'],[10,20,30]):
   n = 2*m+1
   mass_a = mass*a
   t2d = exact2d.genT2d(n,mass_a)
   t2d = t2d.reshape((n*n,n*n))
   tinv = scipy.linalg.inv(t2d)
   tinv = tinv.reshape((n,n,n,n))
   plt.plot(tinv[m,m][m,range(m,n)],'r'+st,label='n='+str(n))
#plt.legend()
#plt.show()

# Contraction
iprt = 1
mass2 = mass**2
auxbond = 20
m = 5
n = 2*m+1
result = gen2d.initialization(n,mass2,iprt,auxbond)
scale,zpeps,local2,local1a,local1b = result

for auxbond in [20]:
   print '\ncase-d: right'
   vlst = []
   pa = (m,m)
   for i in range(1,np):
      pb = (m,m+i)
      epeps = zpeps.copy()
      epeps[pa] = local1a
      epeps[pb] = local1b
      for j in range(m,m+i):
         epeps[m,j] = numpy.einsum('ludr,u->ludr',epeps[m,j],[1,-1,-1,1])
         val = contraction2d.contract(epeps,auxbond)
      print ' points=',pa,pb,'auxbond=',auxbond,'val=',val
      vlst.append(val)
   plt.plot(range(1,np),vlst,'gs-',label='CPEPS n='+str(n))

# Test CTM
import ctm
m = 5
n = 2*m+1
nsteps = 30
nt = n+2*nsteps
chi = 40
for auxbond in [20]:
   zpeps0 = ctm.formPEPS1(n,zpeps,chi,nsteps)
   scale,z = contraction2d.binarySearch(zpeps0,chi,iprt=iprt)
   print '\ncase-d: right - TEST-CTM (n,nt)=',(n,nt),' chi=',chi
   vlst = []
   pa = (m,m)
   for i in range(1,np):
      pb = (m,m+i)
      epeps = scale*zpeps0.copy()
      epeps[pa] = scale*local1a
      epeps[pb] = scale*local1b
      for j in range(m,m+i):
         epeps[m,j] = numpy.einsum('ludr,u->ludr',epeps[m,j],[1,-1,-1,1])
      val = contraction2d.contract(epeps,auxbond)
      print ' points=',pa,pb,'auxbond=',auxbond,'val=',val
      vlst.append(val)
   plt.plot(range(1,np),vlst,'ks-',label='CTM: (n,nt)=(%3d,%3d)'%(n,nt))

plt.xlim([-1,20])
plt.legend()
plt.show()
exit()
