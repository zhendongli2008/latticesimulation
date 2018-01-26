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

