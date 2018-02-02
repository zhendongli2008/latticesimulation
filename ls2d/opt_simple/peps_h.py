import peps
import copy
from include import np as np
from include import npeinsum as einsum

I = np.eye(2)
Sz = .5*np.array([[1.,0.],[0.,-1.]])
Sm = np.array([[0.,1.],[0.,0.]])
Sp = Sm.T

def eval_hbond(pepsa0, pepsa, i, j, auxbond):
    pepsb = peps.copy(pepsa)
    pepsb[i,j] = einsum("pq,qludr->pludr",Sz, pepsa[i,j])
    pepsb[i,j+1] = einsum("pq,qludr->pludr",Sz, pepsa[i,j+1])
    valzz = peps.dot(pepsa0,pepsb,auxbond)

    pepsb = peps.copy(pepsa)
    pepsb[i,j] = einsum("pq,qludr->pludr",Sp, pepsa[i,j])
    pepsb[i,j+1] = einsum("pq,qludr->pludr",Sm, pepsa[i,j+1])
    valpm = peps.dot(pepsa0,pepsb,auxbond)

    pepsb = peps.copy(pepsa)
    pepsb[i,j] = einsum("pq,qludr->pludr",Sm, pepsa[i,j])
    pepsb[i,j+1] = einsum("pq,qludr->pludr",Sp, pepsa[i,j+1])
    valmp = peps.dot(pepsa0,pepsb,auxbond)
    return valzz + .5*(valpm+valmp)


def eval_vbond(pepsa0, pepsa, i, j,auxbond):
    pepsb = peps.copy(pepsa)
    pepsb[i,j] = einsum("pq,qludr->pludr",Sz, pepsa[i,j])
    pepsb[i+1,j] = einsum("pq,qludr->pludr",Sz, pepsa[i+1,j])
    valzz = peps.dot(pepsa0,pepsb,auxbond)

    pepsb = peps.copy(pepsa)
    pepsb[i,j] = einsum("pq,qludr->pludr",Sp, pepsa[i,j])
    pepsb[i+1,j] = einsum("pq,qludr->pludr",Sm, pepsa[i+1,j])
    valpm = peps.dot(pepsa0,pepsb,auxbond)
    
    pepsb = peps.copy(pepsa)
    pepsb[i,j] = einsum("pq,qludr->pludr",Sm, pepsa[i,j])
    pepsb[i+1,j] = einsum("pq,qludr->pludr",Sp, pepsa[i+1,j])
    valmp = peps.dot(pepsa0,pepsb,auxbond)
    return valzz + .5*(valpm+valmp)


def eval_heish(pepsa, pepsb, auxbond=None):
    shape = pepsa.shape
    nr,nc=shape
    val=0.
    for i in range(nr):
        for j in range(nc-1):
            val += eval_hbond(pepsa,pepsb,i,j,auxbond)
    for i in range(nr-1):
        for j in range(nc):
            val += eval_vbond(pepsa,pepsb,i,j,auxbond)
    return val
