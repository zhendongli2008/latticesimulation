#
# Convention: pludr
#
import mps
import mpo
from include import np
from include import npeinsum as einsum

def copy(pepsa):
    shape = pepsa.shape
    pepsb = np.empty(shape, dtype=np.object)
    for i in range(shape[0]):
       for j in range(shape[1]):
	  pepsb[i,j] = pepsa[i,j]
    return pepsb

def zeros(shape,pdim,bond):
    peps = empty(shape,pdim,bond)
    for i in range(peps.shape[0]):
        for j in range(peps.shape[1]):
            peps[i,j] = np.zeros_like(peps[i,j])
    return peps

def random(shape,pdim,bond,fac=0.1):
    peps = empty(shape,pdim,bond)
    for i in range(peps.shape[0]):
       for j in range(peps.shape[1]):
          peps[i,j] = np.random.uniform(-1.,1.,peps[i,j].shape)*fac
    return peps
    
def empty(shape,pdim,bond):
    peps = np.zeros(shape, dtype=np.object)
    # dimension of bonds, ludr
    ldims=np.ones(shape[1],dtype=np.int)*bond
    ldims[0]=1
    rdims=np.ones(shape[1],dtype=np.int)*bond
    rdims[-1]=1
    ddims=np.ones(shape[0],dtype=np.int)*bond
    ddims[0]=1
    udims=np.ones(shape[0],dtype=np.int)*bond
    udims[-1]=1
    for i in range(shape[0]):
        for j in range(shape[1]):
            peps[i,j] = np.empty([pdim,ldims[j],udims[i],ddims[i],rdims[j]])
    return peps

def ceval(peps,config,auxbond):
    shape = peps.shape
    cpeps=np.zeros(shape, dtype=np.object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            cpeps[i,j]=peps[i,j][config[i,j],:,:,:,:]
    return contract_cpeps(cpeps,auxbond)

def cpeps(peps,config):
    shape = peps.shape
    cpeps=np.empty(shape, dtype=np.object)
    peps_config = np.reshape(config, peps.shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            cpeps[i,j]=peps[i,j][peps_config[i,j],:,:,:,:]
    return cpeps

def epeps(pepsa,pepsb):
    shape = pepsa.shape
    epeps = np.empty(shape, dtype=np.object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            epeps[i,j]=einsum("pludr,pLUDR->lLuUdDrR",pepsa[i,j],pepsb[i,j])
            eshape=epeps[i,j].shape
            epeps[i,j]=np.reshape(epeps[i,j],(eshape[0]*eshape[1],
                                              eshape[2]*eshape[3],
                                              eshape[4]*eshape[5],
                                              eshape[6]*eshape[7]))
    return epeps

def add(pepsa,pepsb):
    pdim = pepsa[0,0].shape[0]
    bonda = pepsa[0,0].shape[-1] # right bond
    bondb = pepsb[0,0].shape[-1] # right bond
    shape = pepsa.shape
    pepsc = zeros(shape,pdim,bonda+bondb)
    for i in range(shape[0]):
        for j in range(shape[1]):
            n,la,ua,da,ra = pepsa[i,j].shape
            n,lc,uc,dc,rc = pepsc[i,j].shape
            pepsc[i,j][:,:la,:ua,:da,:ra] = pepsa[i,j]
            # pepsb
            l1,l2 = la,lc
            u1,u2 = ua,uc
            d1,d2 = da,dc
            r1,r2 = ra,rc
            # Boundary case
            if i == 0: 	          # first row 
               assert da == dc == 1
               d1,d2 = 0,1
            elif i == shape[0]-1: # last row
               assert ua == uc == 1
               u1,u2 = 0,1
            if j == 0: 	 	  # first col
               assert la == lc == 1
               l1,l2 = 0,1
            if j == shape[1]-1:   # last col
               assert ra == rc == 1
               r1,r2 = 0,1
            pepsc[i,j][:,l1:l2,u1:u2,d1:d2,r1:r2] = pepsb[i,j]
    return pepsc
    
def add_noise(peps,pdim,bond,fac=1.0):
    vec = flatten(peps)
    vec = vec + fac*np.random.uniform(-1.,1.,vec.shape)
    peps_new = aspeps(vec,peps.shape,pdim,bond)
    return peps_new

def create(shape,pdim,config):
    peps_config = np.reshape(config, shape)
    peps0 = zeros(shape,pdim,1)
    for i in range(shape[0]):
        for j in range(shape[1]):
            peps0[i,j][peps_config[i,j],0,0,0,0]=1.
    return peps0
      
def dot(pepsa,pepsb,auxbond):
    epeps0 = epeps(pepsa,pepsb)
    return contract_cpeps(epeps0, auxbond)

def size(peps):
    size=0
    for i in range(peps.shape[0]):
        for j in range(peps.shape[1]):
            size+=peps[i,j].size
    return size
    
def flatten(peps):
    vec=np.empty((0))
    for i in range(peps.shape[0]):
        for j in range(peps.shape[1]):
            vec = np.append(vec,np.ravel(peps[i,j]))

    return vec

def aspeps(vec,shape,pdim,bond):
    peps0 = empty(shape,pdim,bond)
    assert vec.size == size(peps0)
    ptr=0
    for i in range(shape[0]):
        for j in range(shape[1]):
            nelem = peps0[i,j].size
            peps0[i,j] = np.reshape(vec[ptr:ptr+nelem],
                                    peps0[i,j].shape)
            ptr += nelem
    return peps0
    
def contract_cpeps(cpeps,auxbond):
    cmps0 = [None] * cpeps.shape[1]
    for i in range(cpeps.shape[1]):
        l,u,d,r = cpeps[0,i].shape
        cmps0[i] = np.reshape(cpeps[0,i], (l,u*d,r))
    for i in range(1,cpeps.shape[0]):
        cmpo = [None] * cpeps.shape[1]
        for j in range(cpeps.shape[1]):
            cmpo[j] = cpeps[i,j]
        cmps0 = mpo.mapply(cmpo,cmps0)
        if auxbond is not None: # compress
            cmps0 = mps.SVDcompress(cmps0,auxbond)
    return mps.ceval(cmps0, [0]*cpeps.shape[1])
