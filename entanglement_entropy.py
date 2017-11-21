import numpy
import scipy.linalg
import matplotlib.pyplot as plt

# S = \sum_i -si^2*log(si^2)
def vonNeumannEntropy(sigs,thresh=1.e-12):
   ssum = 0.
   for sig2 in sigs:
     if sig2 < thresh: continue
     ssum += -sig2*numpy.log2(sig2)
   return ssum

L = 3000
# -1/2d^2/dx^2
to = numpy.identity(L)
for i in range(L-1):
   to[i,i+1] = to[i+1,i] = -0.5
tp = numpy.identity(L)
for i in range(L-1):
   tp[i,i+1] = tp[i+1,i] = -0.5
tp[0,L-1] = tp[L-1,0] = -0.5

for i in range(L):
   to[i,i] += 1.e-5
for i in range(L):
   tp[i,i] += 1.e-5
   
eo,vo = scipy.linalg.eigh(to)
ep,vp = scipy.linalg.eigh(tp)

plt.plot(vo[:,0],'r-')
plt.plot(vp[:,0],'b-')
plt.plot(vo[:,1],'r-')
plt.plot(vp[:,1],'b-')
plt.plot(vo[:,2],'r-')
plt.plot(vp[:,2],'b-')
plt.show()

# restricted to 1/3 of the chain
p = 20
nlst = numpy.zeros(p)
olst = numpy.zeros(p)
plst = numpy.zeros(p)
for i in range(1,p+1):
   n = 50*i	
   # half-filling state
   frag = vo[(L-n)/2:(L+n)/2,:L/2]
   so = scipy.linalg.svd(frag,compute_uv=False)
   frag = vp[(L-n)/2:(L+n)/2,:L/2]
   sp = scipy.linalg.svd(frag,compute_uv=False)
   # |Psi> = Prod_i (si|fi>+sqrt(1-si)|bi>)
   svon_o = vonNeumannEntropy(so**2,thresh=1.e-20)
   svon_p = vonNeumannEntropy(sp**2,thresh=1.e-20)
   nlst[i-1] = n
   olst[i-1] = svon_o
   plst[i-1] = svon_p

plt.plot(nlst,olst,'ro-')
plt.plot(nlst,plst,'bo-')

#
# Fitting
#
from scipy.optimize import curve_fit
def func(x,a,b):
    return a*numpy.log2(x)+b

popt, pcov = curve_fit(func,nlst,olst)
a = popt[0]
b = popt[1]
tlst = a*numpy.log2(nlst)+b
plt.plot(nlst,tlst,'r--')
print 'a_o=',a
popt, pcov = curve_fit(func,nlst,plst)
a = popt[0]
b = popt[1]
tlst = a*numpy.log2(nlst)+b
plt.plot(nlst,tlst,'b--')
print 'a_p=',a
plt.show()

plt.plot(nlst,[2.0**i for i in olst],'ro-')
plt.plot(nlst,[2.0**i for i in plst],'bo-')
plt.show()
