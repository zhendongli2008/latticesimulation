import utils
import copy
from include import np as N
from include import svd as svd
from include import iop as svd_iop
import scipy.linalg

def check_lortho(tens):
    tensm=N.reshape(tens,[N.prod(tens.shape[:-1]),tens.shape[-1]])
    s=N.dot(N.conj(tensm.T),tensm)
    return scipy.linalg.norm(s-N.eye(s.shape[0]))

def check_rortho(tens):
    tensm=N.reshape(tens,[tens.shape[0],N.prod(tens.shape[1:])])
    s=N.dot(tensm,N.conj(tensm.T))
    return scipy.linalg.norm(s-N.eye(s.shape[0]))

def conj(mps):
    """
    complex conjugate
    """
    return [N.conj(mt) for mt in mps]

def is_left_canonical(mps,thresh=1.e-8):
    ret=True
    for mt in mps[:-1]:
        #print check_lortho(mt)
        ret*=check_lortho(mt)<thresh
    return ret

def is_right_canonical(mps,thresh=1.e-8):
    ret=True
    for mt in mps[1:]:
        #print check_rortho(mt)
        ret*=check_rortho(mt)<thresh
    return ret

def shape(mps):
    """
    shapes of tensors
    """
    return [mt.shape for mt in mps]

def zeros(nsites,pdim,m):
    mps=[None]*nsites
    mps[0]=np.zeros([1,pdim,m])
    mps[-1]=np.zeros([m,pdim,1])
    for i in xrange(1,nsites-1):
        mps[i]=np.zeros([m,pdim,m])
    return mps
    
def random(nsites,pdim,m):
    """
    create random MPS for nsites, with m states,
    and physical dimension pdim
    """
    mps=[None]*nsites
    mps[0]=np.random.random([1,pdim,m])
    mps[-1]=np.random.random([m,pdim,1])
    for i in xrange(1,nsites-1):
        mps[i]=np.random.random([m,pdim,m])
    return mps

def ceval(mps,config):
    """
    Evaluates mps at given config
    """
    mps_mats=[None]*len(config)
    nsites=len(config)
    for i, pval in enumerate(config):
        mps_mats[i]=mps[i][:,pval,:]
    
    # multiply "backwards" from right to left
    val=mps_mats[0]
    for i in xrange(1,nsites):
        val=N.dot(val,mps_mats[i])

    # turn into scalar
    return N.trace(val)

def create(pdim,config):
    """
    Create dim=1 MPS
    pdim: physical dimension
    """
    nsites=len(config)
    mps=[utils.zeros([1,pdim,1]) for i in xrange(nsites)]
    for i,p in enumerate(config):
        mps[i][0,p,0]=1.
    return mps
        
def canonicalise(mps,side):
    """
    create canonical MPS
    """
    if side=='l':
        return compress(mps,'r',0)
    else:
        return compress(mps,'l',0)

# SVD-based: ZL@20180118
def SVDcompress(mps,auxbond):
   nmps = compress(mps,"l",trunc=0.0)
   nmps = compress(nmps,"r",trunc=auxbond)
   nmps = compress(nmps,"l",trunc=auxbond)
   return nmps

def compress(mps,side,trunc=1.e-12,check_canonical=False):
    """
    inp: canonicalise MPS (or MPO)

    trunc=0: just canonicalise
    0<trunc<1: sigma threshold
    trunc>1: number of renormalised vectors to keep

    side='l': compress LEFT-canonicalised MPS 
              by sweeping from RIGHT to LEFT
              output MPS is right canonicalised i.e. CRRR

    side='r': reverse of 'l'
   
    returns:
         truncated or canonicalised MPS
    """
    assert side in ["l","r"]

    # if trunc==0, we are just doing a canonicalisation,
    # so skip check, otherwise, ensure mps is canonicalised
    if trunc != 0 and check_canonical:
        if side=="l":
            assert is_left_canonical(mps)
        else:
            assert is_right_canonical(mps)

    ret_mps=[]
    nsites=len(mps)

    if side=="l":
        res=mps[-1]
    else:
        res=mps[0]

    for i in xrange(1,nsites):
        # physical indices exclude first and last indices
        pdim=list(res.shape[1:-1])

        if side=="l":
            res=N.reshape(res,(res.shape[0],N.prod(res.shape[1:])))
        else:
            res=N.reshape(res,(N.prod(res.shape[:-1]),res.shape[-1]))

        if svd_iop == 0:
	   u,sigma,vt=svd(res, full_matrices=False, lapack_driver='gesvd')
        else:
           u,sigma,vt=svd(res,full_matrices=False)

        if trunc==0:
            m_trunc=len(sigma)
        elif trunc<1.:
            # count how many sing vals < trunc            
            normed_sigma=sigma/scipy.linalg.norm(sigma)
            m_trunc=len([s for s in normed_sigma if s >trunc])
        else:
            m_trunc=int(trunc)
            m_trunc=min(m_trunc,len(sigma))

        u=u[:,0:m_trunc]
        sigma=N.diag(sigma[0:m_trunc])
        vt=vt[0:m_trunc,:]

        if side=="l":
            u=N.dot(u,sigma)
            res=N.dot(mps[nsites-i-1],u)
            ret_mpsi=N.reshape(vt,[m_trunc]+pdim+[vt.shape[1]/N.prod(pdim)])
        else:
            vt=N.dot(sigma,vt)
            res=N.tensordot(vt,mps[i],1)
            ret_mpsi=N.reshape(u,[u.shape[0]/N.prod(pdim)]+pdim+[m_trunc])
                
        ret_mps.append(ret_mpsi)

    ret_mps.append(res)
    if side=="l":
        ret_mps.reverse()

    #fidelity = dot(ret_mps, mps)/dot(mps, mps)
    #print "compression fidelity:: ", fidelity
    return ret_mps

def scale(mps,val):
    """
    Multiply MPS by scalar
    """
    ret=[mt.copy() for mt in mps]
    ret[-1]*=val
    return ret

def add(mpsa,mpsb):
    """
    add two mps
    """
    if mpsa==None:
        return [mt.copy() for mt in mpsb]
    elif mpsb==None:
        return [mt.copy() for mt in mpsa]

    assert len(mpsa)==len(mpsb)
    nsites=len(mpsa)
    pdim=mpsa[0].shape[1]
    assert pdim==mpsb[0].shape[1]

    mpsab=[None]*nsites
    
    mpsab[0]=N.dstack([mpsa[0],mpsb[0]])
    for i in xrange(1,nsites-1):
        mta=mpsa[i]
        mtb=mpsb[i]
        mpsab[i]=utils.zeros([mta.shape[0]+mtb.shape[0],pdim,
                              mta.shape[2]+mtb.shape[2]])
        mpsab[i][:mta.shape[0],:,:mta.shape[2]]=mta[:,:,:]
        mpsab[i][mta.shape[0]:,:,mta.shape[2]:]=mtb[:,:,:]

    mpsab[-1]=N.vstack([mpsa[-1],mpsb[-1]])
    return mpsab
        
def dot(mpsa,mpsb):
    """
    dot product of two mps
    """
    assert len(mpsa)==len(mpsb)
    nsites=len(mpsa)
    e0=N.eye(1,1)
    for i in xrange(nsites):
        # sum_x e0[:,x].m[x,:,:]
        e0=N.tensordot(e0,mpsb[i],1)
        # sum_ij e0[i,p,:] mpsa[i,p,:]
        # note, need to flip a (:) index onto top,
        # therefore take transpose
        e0=N.tensordot(e0,mpsa[i],([0,1],[0,1])).T
    return e0[0,0]

def distance(mpsa,mpsb):
    """
    ||mpsa-mpsb||
    """
    return dot(mpsa,mpsa)-2*dot(mpsa,mpsb)+dot(mpsb,mpsb)
