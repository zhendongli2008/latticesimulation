import numpy as np
import mpnum
from latticesimulation.ls2d.opt_simple import mps

def contract_cpeps(cpeps, auxbond):
    cmps0 = [None] * cpeps.shape[1]
    for i in range(cpeps.shape[1]):
        l,u,d,r = cpeps[0,i].shape
        cmps0[i] = np.reshape(cpeps[0,i], (l,u*d,r))
    for i in range(1,cpeps.shape[0]):
        cmpo = [None] * cpeps.shape[1]
        for j in range(cpeps.shape[1]):
            cmpo[j] = cpeps[i,j]
        cmps0 = mapply(cmpo,cmps0)
        if find_max_D(cmps0) > auxbond: # compress
            cmps0 = compress(cmps0,auxbond)
    return mps.ceval(cmps0, [0]*cpeps.shape[1])

def mapply(mpo, mps):
    """
    apply mpo to mps, or apply mpo to mpo
    """
    nsites=len(mpo)
    assert len(mps)==nsites

    ret=[None]*nsites

    if len(mps[0].shape)==3:
        # mpo x mps
        for i in xrange(nsites):
            assert mpo[i].shape[2]==mps[i].shape[1]
            #mt=N.einsum("apqb,cqd->acpbd",mpo[i],mps[i])
            mt = np.tensordot(mpo[i], mps[i], [[2],[1]])
	    mt = np.einsum('apbcd->acpbd', mt)
            mt=np.reshape(mt,[mpo[i].shape[0]*mps[i].shape[0],mpo[i].shape[1],
                             mpo[i].shape[-1]*mps[i].shape[-1]])
            ret[i]=mt
    elif len(mps[0].shape)==4:
        # mpo x mpo
        for i in xrange(nsites):
            assert mpo[i].shape[2]==mps[i].shape[1]
            #mt=N.einsum("apqb,cqrd->acprbd",mpo[i],mps[i])
	    mt = np.tensordot(mpo[i], mps[i], [[2],[1]])
	    mt = np.einsum('apbcrd->acprbd', mt)
            mt=np.reshape(mt,[mpo[i].shape[0]*mps[i].shape[0],
                             mpo[i].shape[1],mps[i].shape[2],
                             mpo[i].shape[-1]*mps[i].shape[-1]])
            ret[i]=mt

    return ret

def compress(cmps0, maxd, iprnt=0):
    tmp = mpnum.MPArray(cmps0)
    overlap = tmp.compress(method='svd', rank=maxd, relerr=0.0)
    if iprnt == 1:
      val = overlap / mps.dot(cmps0, cmps0)
      print "normalized overlap of compression: ", val
    nsites = len(tmp)
    cmps = np.array([None]*nsites, dtype=object)
    for i in xrange(nsites):
      cmps[i] = tmp.lt[i]
    return cmps

def find_max_D(mps):
    L = len(mps)
    D = 0
    for i in xrange(L):
        if mps[i].shape[0] > D: D = mps[i].shape[0]
        if mps[i].shape[2] > D: D = mps[i].shape[2]
    return D
