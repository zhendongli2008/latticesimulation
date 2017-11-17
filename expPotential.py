#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import scipy.linalg
from pyscf import gto
from pyscf import scf,fci,cc
from pyscf import ao2mo

mol = gto.M()
mol.incore_anyway = True

# Settings
ne = 1
Z = 1.0
a = 0.1
L = 6 # one sided [-L,L]
nc = int(L/a)
n = 2*nc+1
print 'a,L,n,nc=',(a,L,n,nc)
t = -0.5/a**2
x = numpy.linspace(-L,L,num=n,endpoint=True)
print 'x=',len(x),x

def get_v(x):
   A = 1.071295
   k = 1.0/2.385345 # eq(7) of prb91,235151(2015)
   return A*numpy.exp(-k*abs(x))

h1 = numpy.zeros((n,n))
for i in range(n-1):
   h1[i,i+1] = h1[i+1,i] = t  
for i in range(n):
   h1[i,i] = 1.0/a**2 + (-Z)*get_v((i-nc)*a)

#import scipy.linalg
#e,v = scipy.linalg.eigh(h1)
#print e
#print v[:,0]
#exit()

eri = numpy.zeros((n,n,n,n))
for i in range(n):
   for j in range(n):
      #(ii|jj)
      eri[i,i,j,j] = get_v(abs(i-j)*a)

mol.verbose = 5
mol.nelectron = ne
mol.spin = ne%2

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)
# ao2mo.restore(8, eri, n) to get 8-fold symmetry of the integrals
mf._eri = ao2mo.restore(8, eri, n)
e = mf.scf()

c0 = mf.mo_coeff[:,0]
c1 = mf.mo_coeff[:,1]
c2 = mf.mo_coeff[:,2]
import matplotlib.pyplot as plt
plt.plot(x,c0,'ro-')
plt.plot(x,c1,'bo-')
plt.plot(x,c2,'go-')
plt.show()
exit()

# CCSD
mycc = cc.CCSD(mf).run()
dm1 = mycc.make_rdm1()
print numpy.linalg.norm(dm1-dm1.T)

# FCI
fx = fci.FCI(mf)
e,vec = fx.kernel(verbose=5)
print 'eFCI=',e
rdm1 = fx.make_rdm1(vec,n,ne)
print rdm1
