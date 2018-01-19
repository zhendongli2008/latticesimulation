import autograd
import autograd.numpy as np
import copy
from latticesimulation.ls2d.opt_simple import peps

einsum=autograd.numpy.einsum
I = np.eye(2)
Sz = .5*np.array([[1.,0.],[0.,-1.]])
Sm = np.array([[0.,1.],[0.,0.]])
Sp = Sm.T

def eval_sisj(pepsa0, pepsa, i, j, k, l, auxbond):
    pepsb = peps.copy(pepsa)
    pepsb[i,j] = einsum("pq,qludr->pludr",Sz, pepsa[i,j])
    pepsb[k,l] = einsum("pq,qludr->pludr",Sz, pepsa[k,l])
    valzz = peps.dot(pepsa0,pepsb,auxbond)

    pepsb = peps.copy(pepsa)
    pepsb[i,j] = einsum("pq,qludr->pludr",Sp, pepsa[i,j])
    pepsb[k,l] = einsum("pq,qludr->pludr",Sm, pepsa[k,l])
    valpm = peps.dot(pepsa0,pepsb,auxbond)

    pepsb = peps.copy(pepsa)
    pepsb[i,j] = einsum("pq,qludr->pludr",Sm, pepsa[i,j])
    pepsb[k,l] = einsum("pq,qludr->pludr",Sp, pepsa[k,l])
    valmp = peps.dot(pepsa0,pepsb,auxbond)
    return valzz + .5*(valpm+valmp)

def eval_heish(pepsa, pepsb, auxbond=None):
    nr,nc = pepsa.shape
    val = 0.
    for i in range(nr):
     for j in range(nr):
       ij = i*nr+j
       for k in range(nc):
        for l in range(nc):
 	  kl = k*nc+l
	  if ij>=kl: continue
	  tmp = eval_sisj(pepsa,pepsb,i,j,k,l,auxbond)
	  val += tmp/np.sqrt((1.*i-k)**2+(1.*j-l)**2)
    return val

#def product(pepsa, pepsb, auxbond, x):
#    nr,nc = pepsa.shape
#    val = 0.
#    for i in range(nr):
#     for j in range(nr):
#        pepsx = peps.copy(pepsa)
#        pepsx[i,j] = einsum("pq,qludr->pludr",Sz, pepsa[i,j])
#	pepsx[k,l] = einsum("pq,qludr->pludr",Sz, pepsa[k,l])
#        valzz = peps.dot(pepsa0,pepsb,auxbond)
#
#    pepsb = peps.copy(pepsa)
#    pepsb[i,j] = einsum("pq,qludr->pludr",Sp, pepsa[i,j])
#    pepsb[k,l] = einsum("pq,qludr->pludr",Sm, pepsa[k,l])
#    valpm = peps.dot(pepsa0,pepsb,auxbond)
#
#    pepsb = peps.copy(pepsa)
#    pepsb[i,j] = einsum("pq,qludr->pludr",Sm, pepsa[i,j])
#    pepsb[k,l] = einsum("pq,qludr->pludr",Sp, pepsa[k,l])
#    valmp = peps.dot(pepsa0,pepsb,auxbond)
#    return valzz + .5*(valpm+valmp)
#
#
#
#       ij = i*nr+j
#       for k in range(nc):
#        for l in range(nc):
# 	  kl = k*nc+l
#	  if ij>=kl: continue
#	  tmp = eval_sisj(pepsa,pepsb,i,j,k,l,auxbond)
#	  val += tmp/np.sqrt((1.*i-k)**2+(1.*j-l)**2)
#    return val
