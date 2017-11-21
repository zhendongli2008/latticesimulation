import numpy
import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from zmpo_dmrg.source.mpsmpo import mpo_class 
from zmpo_dmrg.source.mpsmpo import mps_class 
import lattice_helper
import lattice_vij

#=======================================
ne = 1
Z = 1.0
a = 0.1 #0.1
L = 10 #8 # one sided [-L,L]
nc = int(L/a)
n = 2*nc+1
beta = 0.1
tau = 0.0001 #5
nsteps = int(beta/tau)
x = numpy.linspace(-L,L,num=n,endpoint=True)
print 'beta,tau,nsteps=',(beta,tau,nsteps)
print 'a,L,n,nc=',(a,L,n,nc)
print 'x=',len(x),x
#=======================================

# single particle spectra
vne = lattice_vij.getVne(n,a,Z=1.0)
tij = lattice_helper.getTij(n,a)
hij = tij + numpy.diag(vne[::2])
e,v = scipy.linalg.eigh(hij)
print 'eig[:5]=',e[:5]
ifplot = False
if ifplot:
   plt.plot(x,v[:,0],'ro-')
   plt.plot(x,v[:,1],'bo-')
   plt.plot(x,v[:,2],'go-')
   plt.plot(x,v[:,3],'b+-')
   plt.plot(x,v[:,4],'r+-')
   plt.show()
   exit()

prj_empo,prj_ompo,prj_dmpo = lattice_helper.genPmpo(n,a,tau)
hmpo = lattice_helper.genHmpo(n,a,vne[::2])
nmpo = lattice_helper.genNmpo(n)
vmpo = lattice_vij.genEVmpo(n,a,ng=10)

# Exact construction
mps = lattice_helper.prodMPS(n)
#mps0 = lattice_helper.genSmps(v[:,0])
#mps = mps.add(mps0)
mps.normalize()
print '<MPS|1|MPS>=',mps.dot(mps)
print '<MPS|N|MPS>=',mps.dot(nmpo.dotMPS(mps))
print '<MPS|H|MPS>=',mps.dot(hmpo.dotMPS(mps))

if n <= 5:
   tij = hmpo.toMat()
   nij = nmpo.toMat()
   e,v = scipy.linalg.eigh(tij)
   print tij.shape
   print e
   for i in range(v.shape[0]):
      ti = v[:,i].dot(tij.dot(v[:,i]))
      ni = v[:,i].dot(nij.dot(v[:,i]))
      print 'i=',i,'ni=',ni,'ti=',ti
   exit()

# ST1: exp(-tau*V)*exp(-tau*T)
energy = numpy.zeros(nsteps)
mus = numpy.zeros(nsteps)
ne = 2.0
ni = mps.dot(nmpo.dotMPS(mps))
mu = ne-ni
D = 4
for i in range(nsteps):
   print ' x0:In',mps.dot(mps)
   
   mps = prj_empo.dotMPS(mps)
   nm1 = lattice_helper.compress(mps,D)
   
   mps = prj_ompo.dotMPS(mps)
   nm2 = lattice_helper.compress(mps,D)
   
   vi = 1/a**2+vne-mu
   lmpo = lattice_helper.genLmpo(n,tau,vi)
   mps = lmpo.dotMPS(mps)
   nm3 = lattice_helper.compress(mps,D)

   # RK4-like evolution
   tmps = vmpo.dotMPS(mps)
   tmps.mul(-tau)
   mps = mps.add(tmps)
   nm4 = lattice_helper.compress(mps,D)

   ei = mps.dot(hmpo.dotMPS(mps))\
      + mps.dot(vmpo.dotMPS(mps))
   ni = mps.dot(nmpo.dotMPS(mps))
   energy[i] = ei
   mus[i] = mu
   mu = 0.0 #mu-(ni-ne)
  
   # check
   print ' x1:Te',nm1
   print ' x2:To',nm2
   print ' x3:Lc',nm3
   print ' x4:ck',nm4
   print 'n=',n,'i=',i,'e=',ei,'n=',ni,'mu=',mu

# Plot
itime = numpy.arange(nsteps)*tau
plt.plot(itime,energy,'ro-',markersize=2)
plt.xlabel('tau (a.u.)')
plt.show()

#
# Extract all local terms: exp(-tau*Hlocal)
#
# >>>  mps = prj_dmpo.dotMPS(mps)
# >>>  print 'x3',mps.dot(mps)
# >>>  exit()
