import numpy
import scipy.linalg
import matplotlib.pyplot as plt
from zmpo_dmrg.source.mpsmpo.mpo_class import class_mpo
from zmpo_dmrg.source.mpsmpo.mps_class import class_mps

# 1D: Exact interaction is exponential exp(-a*x)/(2a)

alpha = 2.0 	 # exponent - SMALL alpha require large ng !!!

L = 10.		 # lattice length
n = 50		 # nsite
e = L/float(n+1) # lattice spacing

ng = 20		 # npoint - NUMERICAL problem > 30. 

# Quadrature
a = 1.0/(1.0+0.5*alpha**2*e**2)
xts,wts = numpy.polynomial.hermite.hermgauss(ng)
wij = numpy.exp(a*numpy.einsum('i,j->ij',xts,xts))
# Wij = V[i,k]e[k]V[j,k] = A[i,k]A[j,k]
eig,v = scipy.linalg.eigh(wij)
wka = numpy.einsum('ik,k->ik',v,numpy.sqrt(eig))

# OBC
tij = numpy.diag([1.0]*n)
for i in range(n-1):
   tij[i,i+1] = tij[i+1,i] = -0.5*a
fac = numpy.power(numpy.linalg.det(tij),1.0/(2.0*n))/\
      numpy.sqrt(numpy.pi)

# Factor
A0 = fac*numpy.einsum('k,ka->a',wts,wka)
B0 = fac*numpy.einsum('k,ka->a',wts*xts,wka)*numpy.sqrt(e*a)
D0 = fac*numpy.einsum('k,ka->a',wts*xts*xts,wka)*e*a
A1 = fac*numpy.einsum('k,ka,kb->ab',wts,wka,wka)
B1 = fac*numpy.einsum('k,ka,kb->ab',wts*xts,wka,wka)*numpy.sqrt(e*a)
D1 = fac*numpy.einsum('k,ka,kb->ab',wts*xts*xts,wka,wka)*e*a  # 1/2 (diagonal part) is removed.

idn = numpy.identity(2)
nii = numpy.identity(2); nii[0,0] = 0.0
# first [A0,B0,D0]
site0 = numpy.zeros((1,3*ng,2,2))
site0[0,:ng] 	 = numpy.einsum('a,mn->amn',A0,idn)
site0[0,ng:2*ng] = numpy.einsum('a,mn->amn',B0,nii)
site0[0,2*ng:]   = numpy.einsum('a,mn->amn',D0,nii)/2
# last [D0,B0,A0]
site1 = numpy.zeros((3*ng,1,2,2))
site1[:ng,0] = numpy.einsum('a,mn->amn',D0,nii)/2
site1[ng:2*ng,0] = numpy.einsum('a,mn->amn',B0,nii)
site1[2*ng:,0] = numpy.einsum('a,mn->amn',A0,idn)
# centeral
# [A1,B1,D1]
# [ 0,A1,B1]
# [ 0, 0,A1]
site2 = numpy.zeros((3*ng,3*ng,2,2))
site2[:ng,:ng] = numpy.einsum('ab,mn->abmn',A1,idn)
site2[ng:2*ng,ng:2*ng] = numpy.einsum('ab,mn->abmn',A1,idn)
site2[2*ng:,2*ng:] = numpy.einsum('ab,mn->abmn',A1,idn)
site2[:ng,ng:2*ng] = numpy.einsum('ab,mn->abmn',B1,nii)
site2[ng:2*ng,2*ng:] = numpy.einsum('ab,mn->abmn',B1,nii)
site2[:ng,2*ng:] = numpy.einsum('ab,mn->abmn',D1,nii)/2
sites = [site0]+[site2]*(n-2)+[site1]
mpo = class_mpo(n,sites)

# Simpler MPO:
# [1 sqrt(a)    1 ]
# [0    a  sqrt(a)]  a=exp(-alpha*e)/(2*alpha)**(1/n)
# [0    0       1 ]
afac = numpy.exp(-alpha*e)
site0 = numpy.zeros((1,3,2,2))
site0[0,0] = idn
site0[0,1] = numpy.sqrt(afac)*nii
site0[0,2] = nii/2.0
site1 = numpy.zeros((3,1,2,2))
site1[0,0] = nii/2.0
site1[1,0] = numpy.sqrt(afac)*nii
site1[2,0] = idn
site2 = numpy.zeros((3,3,2,2))
site2[0,0] = idn
site2[0,1] = numpy.sqrt(afac)*nii
site2[0,2] = nii/2.0
site2[1,1] = afac*idn
site2[1,2] = numpy.sqrt(afac)*nii
site2[2,2] = idn
fac = numpy.power(1.0/(2.0*alpha),1.0/float(n))
sites = [fac*site0]+[fac*site2]*(n-2)+[fac*site1]
mpo = class_mpo(n,sites)

def fromMPOMPS(i,j):
   occ = numpy.zeros(n)
   occ[i-1] = 1.0
   occ[j-1] = 1.0
   mps = class_mps(n)
   mps.hfstate(n,occ)
   val = mps.dot(mpo.dotMPS(mps))
   return val

# V = 1/2*ni*Vij*nj (= 1/2*V11+1/2*V22+V12; = 1/2*V11)
def genVij(i,j):
   if i == j:
      val = 2.0*fromMPOMPS(i,i)
   else:
      val = fromMPOMPS(i,j)\
	  - fromMPOMPS(i,i)\
	  - fromMPOMPS(j,j)
   return val

print fromMPOMPS(n-1,n-1)
print fromMPOMPS(0,1)
print fromMPOMPS(1,n-1)
print fromMPOMPS(2,3)
print fromMPOMPS(2,5)
print fromMPOMPS(2,50)

#mpo.compress()
#mpo.prt()
#print 3*ng

#=====

def genVal(i,j):
   wfac = [A0]+[A1]*(n-2)+[A0]
   if i == j:
      if i == 1 or i == n:
	 wfac[i-1] = D0
      else:
         wfac[i-1] = D1
   elif i != j:
      if i == 1 or i == n: 
	 wfac[i-1] = B0
      else:
	 wfac[i-1] = B1
      if j == 1 or j == n: 
	 wfac[j-1] = B0
      else:
	 wfac[j-1] = B1
   return reduce(numpy.dot,wfac)

print '\ndata'
print genVal(0,0)
# Another question: what will be the on-site interaction
print genVal(n-1,n-1)
print genVal(0,1)
print genVal(1,n-1)
print genVal(2,3)
print genVal(2,5)

print '\nComparison:'
print genVal(2,3)
print fromMPOMPS(2,3)
print fromMPOMPS(2,2)
print fromMPOMPS(3,3)
print genVij(2,3)

def genExt(i,j):
   return numpy.exp(-alpha*e*abs(i-j))/(2.0*alpha)

def genPro(i,j):
   x = e*min(i,j)
   y = e*max(i,j)
   return numpy.sinh(alpha*x)*\
   	  numpy.sinh(alpha*(L-y))/\
	  (alpha*numpy.sinh(alpha*L))

xlst = numpy.arange(1,n+1)*e

# Vii
mlst0 = [genVij(i,i) for i in range(1,n+1)]
ylst0 = [genVal(i,i) for i in range(1,n+1)]
ylst1 = [genPro(i,i) for i in range(1,n+1)]
ylst2 = [genExt(i,i) for i in range(1,n+1)]

plt.plot(xlst,mlst0,'g+-',label='MPO')
plt.plot(xlst,ylst0,'k--',label='Num')
plt.plot(xlst,ylst1,'b--',label='OBC')
plt.plot(xlst,ylst2,'r--',label='Exp')
plt.xlim(0,L+0.1)
plt.ylim(0.0,1/(2*alpha)+0.1)
#plt.legend()
#plt.show()

# Vij
mlst3 = [genVij(int(5),i) for i in range(1,n+1)]
ylst1 = [genVal(int(5),i) for i in range(1,n+1)]
ylst2 = [genVal(int(n/6.0),i) for i in range(1,n+1)]
ylst3 = [genVal(int(n/3.0),i) for i in range(1,n+1)]
ylst4 = [genVal(int(n/2.0),i) for i in range(1,n+1)]
mlst4 = [genPro(int(n/2.0),i) for i in range(1,n+1)]
nlst4 = [genExt(int(n/2.0),i) for i in range(1,n+1)]

plt.plot(xlst,mlst3,'g-',markersize=10)
plt.plot(xlst,ylst1,'k+-',markersize=10)
plt.plot(xlst,ylst2,'k+-',markersize=10)
plt.plot(xlst,ylst3,'k+-',markersize=10)
plt.plot(xlst,ylst4,'k+-',markersize=10)
plt.plot(xlst,mlst4,'ro-',markersize=2)
plt.plot(xlst,nlst4,'b--',markersize=2)
plt.legend()
plt.savefig('fig_a'+str(alpha)+'_n'+str(n)+'_g'+str(ng)+'.pdf')
plt.show()
