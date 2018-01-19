import numpy
from ising_corr import ising_localtensor_fast as lt
import mpnum
import copy

DEBUG = False
MAX_D = None # defined in "if name == main" block
tensordot = numpy.tensordot

# start by importing / setting up ising grid with out-of-plane indices at
# the corrects points, then contract ising grid and phys grid together

##################### NEED TO DEFINE NI,NJ ####################################
#Ni = [[1,0],[0,-1]]
#Nj = [[1,0],[0,-1]]
##############################################################


# populate the tensors that belong on the physical sites.
# the index numbers are labelled as follows:
# "0" means "exterior tensor"
# "1" is used to limit the contractions to only two-body terms
# "2" means "on interaction path in typical case"
# "3" means "on interaction path in symmetrical case"
# INDEX CONVENTION: T[up, right, down, left, out-of-plane]
def populate_phys(dup, dright, ddown, dleft):
    ########### NEED TO DEFINE Ni, Nj
    Ni = 3.0
    Nj = 3.0
    ################################
    T = numpy.zeros((dup, dright, ddown, dleft, 2))
    T[0,0,0,0,0] = 1

    # bulk tensor
    if (dup != 1 and dright != 1 and ddown != 1 and dleft != 1):
        T[2,0,2,0,0] = T[1,2,0,2,0] = 1 # transmit "interacting" signal
        T[1,0,1,0,0] = T[1,1,0,1,0] = 1 # transmit limiter signal

        T[2,0,0,0,1] = T[1,2,0,0,1] = Ni
        T[1,2,2,0,0] = 1
        T[1,1,2,0,1] = T[1,1,0,2,1] = Nj # these three generate "typical" interactions

        T[1,3,0,3,0] = 1
        T[1,1,2,3,0] = 1
        T[1,3,0,0,1] = Nj # these three account for the "symmetrical" cases

        return T

    # bottom left corner tensor
    elif (ddown == 1 and dleft == 1):
        T[2,0,0,0,1] = T[1,2,0,0,1] = Ni
        return T

    # bottom edge tensor
    elif (ddown == 1 and dright != 1 and dleft != 1):
        T[1,2,0,2,0] = 1
        T[1,1,0,1,0] = 1
        T[2,0,0,0,1] = T[1,2,0,0,1] = Ni
        T[1,1,0,2,1] = Nj

        return T

    # bottom right corner tensor
    elif (ddown == 1 and dright == 1):
        T[1,0,0,1,0] = 1
        T[2,0,0,0,1] = Ni
        T[1,0,0,2,1] = Nj

        return T

    # left edge tensor
    elif (dleft == 1 and dup != 1 and ddown !=1):
        T[2,0,2,0,0] = 1
        T[1,0,1,0,0] = 1
        T[2,0,0,0,1] = T[1,2,0,0,1] = Ni
        T[1,2,2,0,0] = 1
        T[1,1,2,0,1] = Nj
        T[1,3,0,0,1] = Nj

        return T

    # right edge tensor
    elif (dright == 1 and dup != 1 and ddown !=1):
        T[2,0,2,0,0] = 1
        T[1,0,0,1,0] = T[1,0,1,0,0] = 1
        T[2,0,0,0,1] = Ni
        T[1,0,2,0,1] = T[1,0,0,2,1] = Nj
        T[1,0,2,3,0] = 1

        return T

    # top left corner tensor
    elif (dup == 1 and dleft == 1):
        T[0,0,1,0,0] = 1
        T[0,2,0,0,1] = Ni
        T[0,2,2,0,0] = 1
        T[0,1,2,0,1] = Nj
        T[0,3,0,0,1] = Nj

        return T

    # top edge tensor
    elif (dup == 1 and dleft != 1 and dright != 1):
        T[0,2,0,2,0] = 1
        T[0,1,0,1,0] = T[0,0,1,0,0] = 1
        T[0,2,0,0,1] = Ni
        T[0,2,2,0,0] = 1
        T[0,1,2,0,1] = T[0,1,0,2,1] = Nj
        T[0,3,0,3,0] = 1
        T[0,1,2,3,0] = 1
        T[0,3,0,0,1] = Nj

        return T

    # top right tensor
    elif (dup == 1 and dright == 1):
        T[0,0,0,0,0] = 0
        T[0,0,0,1,0] = T[0,0,1,0,0] = 1
        T[0,0,2,0,1] = T[0,0,0,2,1] = Nj
        T[0,0,2,3,0] = 1

        return T

# create the virtual matrices
# INDEX CONVENTION: M[input, output, out-of-plane]
# whereas the out-of plane needs to transmit information in the case of physical
# sites, these out-of-plane indices are just dummy indices to allow for signal
# transmission through the ising tensors.
def populate_virt_phys(din, dout):
    mat = numpy.zeros((din, dout, 1))
    assert din == 4
    assert dout == 4
    for i in range(4):
        mat[i,i,0] = 1.0
    return mat

# add the virtual matrices into the physical grid
# "num" is the number of virtuals between each adjacent physical site
# Takes in the physical site rows, and outputs the full peps grid including
# the virtual matrices on the bonds. The full grid is a 1D list, of which the
# first entry corresponds to the bottom row and the final entry corresponds to
# the top row.
def add_virts(num, bottom, middle, top):
    # first add the virtuals into the existing rows
    N = len(bottom)
    assert N == len(middle)
    assert N == len(top)
    virt = populate_virt_phys(4,4)
    newbot = numpy.array([None]*(N+num*(N-1)), dtype=object)
    newmid = numpy.array([None]*(N+num*(N-1)), dtype=object)
    newtop = numpy.array([None]*(N+num*(N-1)), dtype=object)
    for i in range(len(newbot)):
        if i%(num+1) == 0:
            newbot[i] = bottom[i/(num+1)]
            newmid[i] = middle[i/(num+1)]
            newtop[i] = top[i/(num+1)]
        else:
            newbot[i] = virt
            newmid[i] = virt
            newtop[i] = virt

    # now add the new rows that correspond to virtual matrices living on the
    # bonds that point up and down
    virt_row = numpy.array([None]*(N+num*(N-1)), dtype=object)
    for i in range(len(virt_row)):
        if i%(num+1) == 0:
            virt_row[i] = virt

    #grid = [None]*(N+num*(N-1))
    grid = numpy.zeros((N+num*(N-1), N+num*(N-1)), dtype=object)
    for i in range(len(grid)):
        grid[0,i] = newbot[i]
    for i in range(1, len(grid)-1):
        for j in range(len(grid[i])):
            if i%(num+1) == 0:
                grid[i,j] = newmid[j]
            else: grid[i,j] = virt_row[j]
    for i in range(len(grid)):
        grid[len(grid)-1, i] = newtop[i]
    assert len(grid[0]) == len(grid[1])
    return grid


# create rows of sites
def make_phys_rows(L, J, temp):

    bulk = populate_phys(4,4,4,4)
    T00 = populate_phys(4,4,1,1)
    Tbot = populate_phys(4,4,1,4)
    T0n = populate_phys(4,1,1,4)
    Tright = populate_phys(4,1,4,4)
    Tleft = populate_phys(4,4,4,1)
    Tn0 = populate_phys(1,4,4,1)
    Ttop = populate_phys(1,4,4,4)
    Tnn = populate_phys(1,1,4,4)

    bottom = numpy.array([None] * L, dtype=object)
    middle = numpy.array([None] * L, dtype=object)
    top = numpy.array([None] * L, dtype=object)

    bottom[0] = T00
    for i in range(1,L-1):
        bottom[i] = Tbot
    bottom[L-1] = T0n

    middle[0] = Tleft
    for i in range(1,L-1):
        middle[i] = bulk
    middle[L-1] = Tright

    top[0] = Tn0
    for i in range(1, L-1):
        top[i] = Ttop
    top[L-1] = Tnn

    return bottom, middle, top

# make the physical gird
def make_phys_grid(L,J,T,num):
    bottom, middle, top = make_phys_rows(L,J,T)
    grid  = add_virts(num, bottom, middle, top)
    return grid


# make the spin rows
# note that the input L here is different than that for the physical grid funcs.
# Here L is the side length of the desired spin grid, which should be larger
# than the side length of the physical grid
# !!!!!!!!!! ASSUMPTION: the spins will not be on the edge tensors of the
# grid because there will be some buffer region to help with boundary effects !!!!!!!!!!!!!!!
# OUTPUT: bottom, middle, top, middle1, middle2...top, middle, bottom are
# the regular spin grid rows without any out-of-plane indices. middle1 is
# the spin rows that correspond to the physical lattice rows that are "dense.""
# middle2 is the spin row that corresponds to the physical lattice rows that are
# "sparse."
def make_spin_rows(L, J, temp, phys_grid, num):
    phyL = len(phys_grid[0])
    assert phyL == len(phys_grid)
    Nphys = (phyL + num) / (num + 1.0)
    assert int(Nphys) == Nphys  # make sure that Nphys is an integer

    offset  = int(numpy.floor((L - phyL) / 2.0))
    bottom, middle, top = lt.make_rows(L,J,temp)
    tmpbot = numpy.array([None]*len(bottom), dtype=object)
    tmpmid = numpy.array([None]*len(middle), dtype=object)
    tmptop = numpy.array([None]*len(top), dtype=object)
    for i in range(len(top)):
        tmpbot[i] = bottom[i]
        tmpmid[i] = middle[i]
        tmptop[i] = top[i]
    bottom = copy.deepcopy(tmpbot)
    middle = copy.deepcopy(tmpmid)
    top = copy.deepcopy(tmptop)
    middle1 = copy.deepcopy(middle)
    middle2 = copy.deepcopy(middle)
    for i in range(offset, phyL + offset):
        for j in range(offset, phyL + offset):
            # "dense row" that contains physical sites and virtual matrices
            if (i-offset) % (num+1) == 0:
                # physical site
                if (j-offset) % (num+1) == 0:
                    assert len(phys_grid[i-offset, j-offset].shape) == 5
                    tmps = lt.gen_tensor(2,2,2,2,J,temp,spin=True)
                    tmp = lt.gen_tensor(2,2,2,2,J,temp,spin=False)
                    newtens = numpy.zeros((2,2,2,2,2))
                    for a in range(2):
                        for b in range(2):
                            for c in range(2):
                                for d in range(2):
                                    newtens[a,b,c,d,0] = tmp[a,b,c,d]
                                    newtens[a,b,c,d,1] = tmps[a,b,c,d]
                    middle1[j] = newtens
                else:
                    assert (j-offset) % (num+1) != 0
                    assert len(phys_grid[i-offset, j-offset].shape) == 3
                    middle1[j] = numpy.expand_dims(middle[j], axis=-1)
            # "sparse row" that contains only virtual matrices that have large spacing
            else:
                assert (i-offset) % (num+1) != 0
                if (j-offset) % (num+1) == 0:
                    middle2[j] = numpy.expand_dims(middle[j], axis=-1)

    return bottom, middle, top, middle1, middle2

# make the full spin grid
def make_spin_grid(L,J,T,phys_grid,num):
    phyL = len(phys_grid[0])
    assert phyL == len(phys_grid)
    Nphys = (phyL + num) / (num + 1.0)
    assert int(Nphys) == Nphys  # make sure that Nphys is an integer

    offset  = int(numpy.floor((L - phyL) / 2.0))
    bottom, middle, top, middle1, middle2 = make_spin_rows(L,J,T, phys_grid, num)
    #grid = [None]*L
    grid = numpy.zeros((L,L), dtype=object)
    for i in range(L):
        grid[0,i] = bottom[i]
    count = 0  # just used to check that we allocate the proper number of middlei
    for i in range(1,L-1):
        for j in range(L):
            if i < offset:
                grid[i,j] = middle[j]
            elif i >= offset and i < (phyL + offset):
                if (i-offset) % (num+1) == 0:
                    grid[i,j] = middle1[j]
                else:
                    grid[i,j] = middle2[j]
                count += 1
            elif i >= (phyL + offset):
                grid[i,j] = middle[j]

    assert phyL == count / L  # just used to check that we allocate the proper number of middlei
    for i in range(L):
        grid[L-1,i] = top[i]
    return grid




# run some routines to try to output a more numerically stable grid.
# also scale the grid so that the contraction of the spins gives the ratio corr/Z
# and not just the corr value (which we then would later have to divide by Z).
#
# shouldn't input a grid that has any out of plane indices (for now)
#
# I think it makes most sense to input the "combined grid" into this function,
# before any contraction of the buffer.
def stabilize(grid, L, J, temp, maxd=MAX_D):
    Lgrid = len(grid)
    Lising = L
    assert type(grid) is numpy.ndarray
    assert grid.shape == (Lgrid,Lgrid)
    assert Lgrid == Lising # as long as "combined grid" is the input grid

    print
    print "############## Accessing Ising code to try to do numerical stabilization..."
    bottomz, middlez, topz = lt.make_rows(Lising, J, temp)
    # make a mini grid like in ising function "compute_corr"
    assert Lising >= 5
    minipeps = [None]*5
    minipeps[0] = bottomz
    minipeps[1] = middlez
    minipeps[2] = middlez
    minipeps[3] = middlez
    minipeps[4] = topz
    # this style of grid formation is risky and deprecated for this code, but
    # it is unfortunately used in the ising code, so we need to create the grid
    # this way to ensure backwards compatibility
    num = lt.contract(minipeps, 5000)
    norm = numpy.power(num, 1.0/5.0)
    scale = 1.0 / norm
    lt.mpo_scale(bottomz, scale)
    lt.mpo_scale(middlez, scale)
    lt.mpo_scale(topz, scale)
    for i in range(Lgrid):
        lt.mpo_scale(grid[i], scale)

    # now we are going to normalize by Z^(1/N) on each site of the lattice.
    # Although this is inaccurate due to large numbers, we don't care very much
    # because the fitting process eliminat these errors as long as the set of
    # correlation functions can still represent 1/r
    Z = lt.makeZ(bottomz, middlez, topz, Lising)
    Z = lt.contract(Z, maxd)
    norm = numpy.power(Z, 1.0/Lgrid)
    scale = 1.0 / norm
    for i in range(Lgrid):
        lt.mpo_scale(grid[i], scale)

    return grid





# compress an mps/mpo using mpnum library
# default setting use svd compression and return a compressed mps/mpo which
# has bond dimension maxd unless it can find a smaller D for which the compresion
# error is 0, then it will return that smaller D
def compress(mps, maxd):
    debug = False
    if mps[0].shape[0] == 1 and mps[0].shape[2] != 1:
        newdim = 0
    elif mps[0].shape[2] == 1 and mps[0].shape[0] != 1:
        newdim = 2
    elif mps[0].shape[2] != 1 and mps[0].shape[0] != 1:
        newdim = None
    else:
        raise ValueError("can't assign a value to newdim")

    if debug:
        for i, (ten, nten) in enumerate(zip(mps[:-1], mps[1:])):
            if ten.shape[1] != nten.shape[3]:
                print i
                print ten.shape[1]
                print nten.shape[3]

    mpa = lt.map_to_mpnum(mps)
    overlap = mpa.compress(method='svd', rank=maxd, relerr=0.0)

    if newdim is not None:
        tmp = copy.deepcopy(mps)
        if newdim == 0:
            for i in range(len(tmp)):
                tmp[i] = numpy.einsum('abcd->cbad', tmp[i])
        print "MPS: normalized overlap of compression: {}".format(
                                                overlap / lt.mpo_dot(tmp, tmp))
    else:
        assert newdim is None
        tmp = lt.map_to_mpnum(mps)
        assert len(mpa) == len(tmp)
        print "MPO: normalized frobenius overlap of compression: {}".format(
                                                overlap / mpnum.inner(tmp, tmp))
        #tester1, tester2 = gen_tester(mpa)
        #print ("MPO: <a|compressed|a> / <a|original|a>: ",
        #    mpnum.sandwich(mpa, tester1, tester2) / mpnum.sandwich(tmp, tester1, tester2))
    ret = lt.map_from_mpnum(mpa, newdim)
    return ret

# THIS IS THE EQUIVALENT CALL TO COMPRESS A COLUMN OF THE GRID AS IF IT IS AN MPS/MPO
# compress an mps/mpo using mpnum library
# default setting use svd compression and return a compressed mps/mpo which
# has bond dimension maxd unless it can find a smaller D for which the compresion
# error is 0, then it will return that smaller D
def compress_vert(mps, maxd):
    debug = False

    if mps[0].shape[1] == 1 and mps[0].shape[3] != 1:
        newdim = 1
    elif mps[0].shape[3] == 1 and mps[0].shape[1] != 1:
        newdim = 3
    elif mps[0].shape[3] != 1 and mps[0].shape[1] != 1:
        newdim = None
    else:
        raise ValueError("can't assign a value to newdim")

    if debug:
        for i, (ten, nten) in enumerate(zip(mps[:-1], mps[1:])):
            if ten.shape[0] != nten.shape[2]:
                print i
                print ten.shape[1]
                print nten.shape[3]

    mpa = map_to_mpnum_vert(mps)
    overlap = mpa.compress(method='svd', rank=maxd, relerr=0.0)

    if newdim is not None:
        tmp = numpy.array([None]*len(mps), dtype=object)
        for i in range(len(mps)):
            if newdim == 3:
                tmp[i] = numpy.einsum('abcd->badc', mps[i])
            elif newdim == 1:
                tmp[i] = numpy.einsum('abcd->dabc', mps[i])
        print "MPS: normalized overlap of compression: {}".format(
                                                    overlap / lt.mpo_dot(tmp, tmp))
    else:
        assert newdim is None
        tmp = map_to_mpnum_vert(mps)
        assert len(mpa) == len(tmp)
        print "MPO: normalized frobenius overlap of compression: {}".format(
                                                overlap / mpnum.inner(tmp, tmp))
    ret = map_from_mpnum_vert(mpa, newdim)
    return ret


# THIS IS A DIFFERENT FUNCTION THAN THE ONE IMPLEMENTED IN THE ISING CODE!
# THIS CODE TAKES IN COLUMNS OF THE GRID AND COMPRESSES THE COLUMN AS THOUGH IT
# WERE AN MPS OR MPO!!!!!!!!!!!!!!1
# take an MPO data structure from this code and map it the the correct
# corrsponding mpnum data structure. If the mpo has dright == 1 or dleft == 1, this
# function will map it to a true MPS in the mpnum data structure. If all 4 indices
# have D != 1, it will be mapped to an MPO in mpnum.
# CONVENTION FOR MPNUM MPOS: will find out eventually
# RETURNS AN MPNUM DATA STRUCTURE
def map_to_mpnum_vert(mpo1):
    assert len(mpo1[1].shape) == 4
    nsites = len(mpo1)
    mpo = copy.deepcopy(mpo1)

    # left leg has dimension 1
    if mpo[1].shape[3] == 1 and mpo[1].shape[1] != 1:
        for i in range(nsites):
            mpo[i] = numpy.squeeze(mpo[i], axis=3)
            mpo[i] = numpy.einsum('abc->cba', mpo[i])

        mpa = mpnum.MPArray(mpo)
        return mpa

    # right leg has dimension 1
    elif mpo[1].shape[1] == 1 and mpo[1].shape[3] != 1:
        for i in range(nsites):
            mpo[i] = numpy.squeeze(mpo[i], axis=1)
            mpo[i] = numpy.einsum('abc->bca', mpo[i])

        mpa = mpnum.MPArray(mpo)
        return mpa

    # we have a real MPO
    elif mpo[1].shape[0] !=1 and mpo[1].shape[2] !=1:
        for i in range(nsites):
            mpo[i] = numpy.einsum('abcd->cbda', mpo[i])

        mpa = mpnum.MPArray(mpo)
        return mpa

    else:
        raise ValueError('both physical indices (left and right) have dimension 1')

# newdim is the dimension to add to get back to pseudo mpo form:
# 0 is top, 2 is down...shouldnt be any other number
def map_from_mpnum_vert(mpa, newdim=3):
    assert newdim in [None,1,3]
    nsites = len(mpa)
    mps = numpy.array([None]*nsites, dtype=object)
    if newdim is None:
        for i in range(nsites):
            mps[i] = numpy.einsum('cbda->abcd', mpa.lt[i])
        return mps
    elif newdim == 3:
        for i in range(nsites):
            mps[i] = numpy.einsum('cba->abc', mpa.lt[i])
            mps[i] = numpy.expand_dims(mps[i], 3)
        return mps

    else:
        for i in range(nsites):
            mps[i] = numpy.einsum('bca->abc', mpa.lt[i])
            mps[i] = numpy.expand_dims(mps[i], 1)

        return mps

# helper function to compute the contraction of the physical grid alone. This
# does the equivalent of mpoXmpo, but when the top row is not a real mpo and
# is just some widely spaced virtual matrices.
def mpoXvirt(mpo1, mpo2):
    nsites = len(mpo2)
    assert len(mpo1) == nsites
    ret = [None]*nsites
    for i in xrange(nsites):
        if i%(num+1) == 0:
            ret[i] = numpy.einsum('abcd,ae->ebcd',mpo1[i], mpo2[i])
        else:
            assert mpo2[i] is None
            ret[i] = mpo1[i]
    return ret

# do the mpoXmpo operation for two rows of physical sites that have virtual
# matrices in between them
def mpoXmpo_virt(mpo1, mpo2):
    nsites = len(mpo2)
    assert nsites == len(mpo1)
    ret = [None]*nsites
    for i in xrange(nsites):
        if i%(num+1) == 0:
            mt = numpy.einsum("abcd,efag->efbcgd",mpo1[i], mpo2[i])
            mt = numpy.reshape(mt, [mpo2[i].shape[0], mpo1[i].shape[1]*mpo2[i].shape[1],
                                   mpo1[i].shape[2], mpo1[i].shape[3]*mpo2[i].shape[3]])
            ret[i] = mt
        else:
            mpo1[i] = mpo1[i].reshape(mpo1[i].shape[0],mpo1[i].shape[1],1)
            mpo2[i] = mpo2[i].reshape(mpo2[i].shape[0],mpo2[i].shape[1],1)
            mt = numpy.einsum('abc,dec->daeb', mpo1[i], mpo2[i])
            mt = mt.reshape(mpo1[i].shape[0] * mpo2[i].shape[0], mpo1[i].shape[1] * mpo2[i].shape[1])
            ret[i] = mt

    return ret



# take a full grid with no physical indices pointing out of plane, contract it
# and return a scalar. This is just for the physical grid, which requires a
# weird contraction scheme due to rows that are just virtual matrices.
# CONVENTIONS: grid[0] returns bottom row, row[0] returns leftmost tensor in the
# row, which has the form T[up, right, down, left]
def contract_phys(peps, maxd=MAX_D):
    tmp = peps[0]
    L = len(peps)
    for i in range(1, L):
        if i%(num+1) != 0:
            tmp = mpoXvirt(tmp, peps[i])
        else:
            tmp = mpoXmpo_virt(tmp, peps[i])
            if tmp[0].shape[1] > maxd:
                #compress
                print len(tmp)
                print tmp[0].shape
                print tmp[1].shape
                mpa = lt.map_to_mpnum(tmp)
                overlap = mpa.compress(method='svd', rank=maxd, relerr=0.0)
                print "normalized overlap of compression (1.0 is good): ", overlap / lt.mpo_dot(tmp, tmp)
                tmp = lt.map_from_mpnum(mpa, 2)

    result = tensordot(tmp[0],tmp[1], [[1],[0]])
    result = result.reshape(tmp[0].shape[3], tmp[1].shape[1])
    count = 0
    for i in range(1, len(tmp)-1):
        if i%(num+1) == 0:
            count = 0
            result = numpy.dot(result, tmp[i+1])
        else:
            if count == num-1:
                result = tensordot(result, tmp[i+1], [[1],[3]]).reshape(result.shape[0],tmp[i+1].shape[1])
            else:
                result = numpy.dot(result, tmp[i+1])
                count += 1

    return numpy.asscalar(result)


# contract along the out-of-plane indices to obtain 1 single tensor network
# from the physical grid and the spin grid
def combine(spin_grid, phys_grid, num):
    L = len(phys_grid)
    assert L == len(phys_grid[0])
    offset = int(numpy.floor((len(spin_grid) - L)/2.0))
    result = copy.deepcopy(spin_grid)
    for i in range(offset, L + offset):
        for j in range(offset, L + offset):
            #"dense" row
            if (i-offset) % (num+1) == 0:
                if (j-offset) % (num+1) == 0:

                    assert len(phys_grid[i-offset, j-offset].shape) == 5
                    assert len(result[i,j].shape) == 5
                    result[i,j] = numpy.einsum('abcde,fghie->afbgchdi',result[i,j],phys_grid[i-offset, j-offset])
                    result[i,j] = result[i,j].reshape(spin_grid[i,j].shape[0]*phys_grid[i-offset, j-offset].shape[0],
                                    spin_grid[i,j].shape[1]*phys_grid[i-offset, j-offset].shape[1],
                                    spin_grid[i,j].shape[2]*phys_grid[i-offset, j-offset].shape[2],
                                    spin_grid[i,j].shape[3]*phys_grid[i-offset, j-offset].shape[3])

                else:
                    assert len(phys_grid[i-offset, j-offset].shape) == 3
                    assert len(result[i,j].shape) == 5
                    result[i,j] = numpy.einsum('abc,defgc->debfga',phys_grid[i-offset, j-offset],result[i,j])
                    result[i,j] = result[i,j].reshape(spin_grid[i,j].shape[0],
                                    spin_grid[i,j].shape[1]*phys_grid[i-offset, j-offset].shape[1],
                                    spin_grid[i,j].shape[2],
                                    spin_grid[i,j].shape[3]*phys_grid[i-offset, j-offset].shape[0])

            #"sparse" row
            else:
                assert (i-offset) % (num+1) != 0
                if (j-offset) % (num+1) == 0:
                    assert len(phys_grid[i-offset, j-offset].shape) == 3
                    assert len(result[i,j].shape) == 5
                    result[i,j] = numpy.einsum('abc,defgc->dbefag', phys_grid[i-offset, j-offset], result[i,j])
                    result[i,j] = result[i,j].reshape(
                                    spin_grid[i,j].shape[0]*phys_grid[i-offset, j-offset].shape[1],
                                    spin_grid[i,j].shape[1],
                                    spin_grid[i,j].shape[2]*phys_grid[i-offset, j-offset].shape[0],
                                    spin_grid[i,j].shape[3])
                else:
                    assert phys_grid[i-offset][j-offset] is None

    return result


# contract the buffer spins to the edge of the physical grid
def contract_edge(grid, offset, len_phys, compression=True, maxd=MAX_D):
    debug = False

    assert offset >= 2
    L = len_phys
    result = numpy.zeros((L,L), dtype=object)
    tmp = copy.deepcopy(grid[0])
    for i in range(1,offset+1):
        tmp = mpoXmpo(tmp, grid[i])
        if compression == True:
            if tmp[len(grid)/2].shape[1] > maxd:
                tmp = compress(tmp, maxd)

    tmp2 = copy.deepcopy(grid[L+offset-1])
    for i in range(L+offset, len(grid)):
        tmp2 = mpoXmpo(tmp2, grid[i])
        if compression == True:
            if tmp2[len(grid)/2].shape[1] > maxd:
                tmp2 = compress(tmp2, maxd)
    tmpgrid = numpy.zeros((L, len(tmp)), dtype=object)
    for i in range(len(tmp)):
        tmpgrid[0,i] = tmp[i]
        tmpgrid[L-1,i] = tmp2[i]
    for i in range(1, L-1):
        for j in range(len(tmp)):
            tmpgrid[i,j] = grid[i+offset, j]
    if debug == True:
        print "offset = {}".format(offset)
        print "shape of original grid: {}".format(grid.shape)
        print "shape of tmpgrid: {}".format(tmpgrid.shape)
        print "....tensor shapes in bottom row of tmpgrid"
        for i in range(len(tmp)):
            print tmp[i].shape
        print "....tensor shapes in top row of tmpgrid"
        for i in range(len(tmp2)):
            print tmp2[i].shape
        #tmpgrid[4,9] = 0.0 #just to test if the values stay in the same spots


    tmp = copy.deepcopy(tmpgrid[:,0])
    for i in range(1, offset+1):
        tmp = mpoXmpo_l2r(tmp, tmpgrid[:,i])
        if compression == True:
            if tmp[len(tmpgrid)/2].shape[2] > maxd:
                tmp = compress_vert(tmp, maxd)

    tmp2 = copy.deepcopy(tmpgrid[:,L+offset-1])
    for i in range(L+offset, len(tmpgrid[0])):
        tmp2 = mpoXmpo_l2r(tmp2, tmpgrid[:,i])
        if compression == True:
            if tmp2[len(tmpgrid)/2].shape[2] > maxd:
                tmp2 = compress_vert(tmp2, maxd)
    for i in range(L):
        result[i,0] = tmp[i]
        result[i, L-1] = tmp2[i]
    for i in range(1,L-1):
        for j in range(L):
            result[j,i] = tmpgrid[j, i+offset]
    if debug == True:
        print "shape of tmpgrid: {}".format(tmpgrid.shape)
        print "shape of result: {}".format(result.shape)
        #print result[4,6] #just to check if the values are in the same spots...
                            # ...should be zero
    return result

# contract out all the fictitious sites so that the final grid is the same size
# as the original physical grid
def contract_fict(grid,num, compression=True, maxd=MAX_D):
    debug = False

    L = len(grid[0])
    assert L == len(grid)
    Nphys = (L + num) / (num + 1.0)
    assert int(Nphys) == Nphys  # make sure that Nphys is an integer
    Nphys = int(Nphys)
    result  = numpy.zeros((Nphys, Nphys), dtype=object)
    tmpgrid = numpy.zeros((Nphys, L), dtype=object)

    # number of fictitious sites between adjacent physical sites is even
    if num % 2 == 0:
        assert num >=2
        tmp = copy.deepcopy(grid[0])
        for i in range(1,(num/2)+1):
            tmp = mpoXmpo(tmp, grid[i])
            if compression == True:
                if tmp[len(grid)/2].shape[1] > maxd:
                    tmp = compress(tmp, maxd)
        for i in range(L):
            tmpgrid[0,i] = tmp[i]

        for i in range(1,Nphys-1):
            loc = (num+1)*i
            tmp = copy.deepcopy(grid[loc-num/2])
            for j in range(loc-num/2+1, loc+num/2 + 1):
                tmp = mpoXmpo(tmp, grid[j])
                if compression == True:
                    if tmp[len(grid)/2].shape[1] > maxd:
                        tmp = compress(tmp, maxd)
            for k in range(len(tmp)):
                tmpgrid[i,k] = tmp[k]

        tmp = copy.deepcopy(grid[(L-1) - num/2])
        for i in range((L-1) - num/2 + 1, L):
            tmp = mpoXmpo(tmp, grid[i])
            if compression == True:
                if tmp[len(grid)/2].shape[1] > maxd:
                    tmp = compress(tmp, maxd)
        for i in range(len(tmp)):
            tmpgrid[Nphys-1, i] = tmp[i]

        if debug == True:
            for i in range(len(tmpgrid)):
                print
                print "row {}:".format(i)
                for j in range(len(tmpgrid[i])):
                    print tmpgrid[i,j].shape

        # hoirzontal multiplication of fictitious sites next
        tmp = copy.deepcopy(tmpgrid[:,0])
        for i in range(1, (num/2)+1):
            tmp = mpoXmpo_l2r(tmp, tmpgrid[:,i])
            if compression == True:
                if tmp[len(tmpgrid)/2].shape[2] > maxd:
                    tmp = compress_vert(tmp, maxd)
        for i in range(Nphys):
            result[i,0] = tmp[i]

        for i in range(1, Nphys-1):
            loc = (num+1)*i
            tmp = copy.deepcopy(tmpgrid[:, loc-num/2])
            for j in range(loc-num/2+1, loc+num/2+1):
                tmp = mpoXmpo_l2r(tmp, tmpgrid[:,j])
                if compression == True:
                    if tmp[len(tmpgrid)/2].shape[2] > maxd:
                        tmp = compress_vert(tmp, maxd)
            for k in range(len(tmp)):
                result[k,i] = tmp[k]

        tmp = copy.deepcopy(tmpgrid[:,(L-1)-num/2])
        for i in range((L-1)-num/2+1, L):
            tmp = mpoXmpo_l2r(tmp, tmpgrid[:,i])
            if compression == True:
                if tmp[len(tmpgrid)/2].shape[2] > maxd:
                    tmp = compress_vert(tmp, maxd)
        for i in range(len(tmp)):
            result[i, Nphys-1] = tmp[i]

        if debug == True:
            for i in range(len(result)):
                print
                print "row {}:".format(i)
                for j in range(len(result[i])):
                    print result[i,j].shape
        return result

    elif num % 2 != 0:
        raise ValueError("currently no implementation for odd number of fict sites" +
                                                        " between phys sites")


if __name__ == "__main__":
    """
    L = 5    # dont go above 6 without compression otherwise RAM will get massacred
    Ni = 77.0
    Nj = 6.0
    J = 1.0
    T=3.5*J
    num = 3 # number of virtual sites in between the real sites. So distance between sites - 1
    #T = 2.26933*J
    phys_grid = make_phys_grid(L,J,T,num)
    phyL = len(phys_grid[0])

    answer = contract_phys(phys_grid)
    print numpy.all(answer == L**2 * (L**2 - 1)/2.0 * numpy.dot(Ni, Nj))
    print answer
    print L**2 * (L**2 - 1)/2.0 * numpy.dot(Ni, Nj)
    """
    print "########## BEGIN TESTING FOR HOW THE GRIDS LOOK"
    MAX_D = 16
    Ni = Nj = 2.0
    num = 4 # this is the correct value, gives 6*6 squares of fict sites for
            # each square of phys sites. Also gives 5*sqrt(2) distance to the
            # nearest diagonal phys sites
    L = 10
    Lphy = L + num*(L-1)
    phys_grid = make_phys_grid(L,1.0,2.5,num)
    print ".........confirm that all rows of physical grid are same length:"
    phyL = len(phys_grid[0])
    assert phyL == len(phys_grid)
    for i in range(1,len(phys_grid)):
        if phyL == len(phys_grid[i]):
            if i == len(phys_grid)-1:
                print "True"
            else:
                continue
        else:
            print "False"
            break


    print "......shape of all tensors in the physical grid"
    for i in range(phyL):
        print
        print "row {}:".format(i)
        for j in range(phyL):
            if phys_grid[i,j] is None:
                print "None"
            else:
                print phys_grid[i,j].shape

    print "........... confirm that all rows of the spin grid are same length:"
    L = 100
    J = 1.0
    temp = 3.0
    spin_grid = make_spin_grid(L,J,temp, phys_grid, num)
    assert L == len(spin_grid)
    assert L == len(spin_grid[0])
    for i in range(1,L):
        if L == len(spin_grid[i]):
            if i == len(spin_grid)-1:
                print "True"
            else:
                continue
        else:
            print "False"
            print "break at iteration {}...".format(i)
            break

    print "........shape of all tensors in the spin grid"
    for i in range(L):
        print
        print "row {}:".format(i)
        for j in range(L):
            if spin_grid[i][j] is None:
                print "None"
            else:
                print spin_grid[i,j].shape


    print "############ TESTING HOW THE COMBINED GRID LOOKS ########"
    full_grid = combine(spin_grid, phys_grid, num)
    assert len(spin_grid) == len(full_grid)
    assert len(spin_grid[0]) == len(full_grid[0])
    for i in range(len(full_grid)):
        print
        print "row {}:".format(i)
        for j in range(len(full_grid[i])):
            print full_grid[i,j].shape

    print "########### TESTING THE STABILIZATION OF COMBINED GRID ##########"
    full_grid = stabilize(full_grid, L, J, temp, maxd=8)

    print "############### BUFFER CONTRACTION TEST ##############"
    offset = int(numpy.floor((L-phyL)/2.0))
    tmpgrid = contract_edge(full_grid, offset, phyL, maxd=MAX_D)
    print "shape of grid: {}".format(tmpgrid.shape)
    print "............shape of tensors in resulting grid"
    for i in range(len(tmpgrid)):
        print
        print "row {}:".format(i)
        for j in range(len(tmpgrid[i])):
            print tmpgrid[i,j].shape

    print "##################### FINAL GRID SHAPE #############"
    final = contract_fict(tmpgrid, num, maxd=MAX_D)
    for i in range(len(final)):
        print
        print "row {}:".format(i)
        for j in range(len(final[i])):
            print final[i,j].shape
