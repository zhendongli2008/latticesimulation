#import autograd.numpy as np
import numpy as np
import autograd
import scipy.optimize
from latticesimulation.ls2d.opt_simple import peps
from latticesimulation.ls2d.opt_simple import peps_h
from latticesimulation.ls2d.lr_test import peps_hlr
import spepo_hlr
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
dtype = np.float_
mtype = MPI.DOUBLE

iop = 3
np.random.seed(0)
nr = 4
nc = 4
pdim = 2
bond = 2
auxbond = 4

# interface to autograd:
def energy1(vec, bond, iprt=0):
   P = peps.aspeps(vec, (nr,nc), pdim, bond)
   PHP = peps_hlr.eval_heish(P, P, auxbond, iop)
   PP = peps.dot(P,P,auxbond)
   e = PHP/PP
   if iprt and rank==0: print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
   return e 
# dlog<P|1+tH+t^2+...|P>/dt|(t=0) = Energy
def energy2(vec, bond, iprt):
   P = peps.aspeps(vec, (nr,nc), pdim, bond)
   PHP = spepo_hlr.eval_heish(P, P, iop)
   PP = peps.dot(P,P,auxbond)
   e = PHP/PP
   if iprt and rank==0: print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
   return e 

energyFun = energy2
bound_energy_fn = lambda x: energyFun(x,bond,1)
deriv = autograd.grad(bound_energy_fn)

def deriv_mpi(vec):
    if rank == 0: print '[deriv_mpi] size=',size
    comm.Barrier()
    t0 = time.time()
    vec_iproc = np.zeros_like(vec)
    if rank == 0: vec_iproc = vec
    comm.Bcast([vec_iproc,mtype])
    ndim = vec.shape[0]
    # Central difference
    eps = 1.e-3
    sgn = [1.,-1.]
    val_iproc = np.zeros_like(vec_iproc) 
    for i in range(2*ndim):
       idr = i//2
       isn = i%2
       if i%size == rank:
          vtmp = vec_iproc.copy()
	  vtmp[idr] += sgn[isn]*eps
	  val_iproc[idr] += sgn[isn]*energyFun(vtmp,bond,0)
    # Reduce
    val = np.zeros_like(val_iproc)
    comm.Reduce([val_iproc,mtype],
		[val,mtype],op=MPI.SUM,root=0)
    val = val/(2.0*eps)
    comm.Bcast([val,mtype])
    t1 = time.time()
    if rank == 0: print '[deriv_mpi] time=',t1-t0
    return val 

def test_min():
    # Initialization
    configa = np.zeros([nr,nc], dtype=np.int)
    configb = np.zeros([nr,nc], dtype=np.int)
    for i in range(nr):
       for j in range(nc):
          configa[i,j] = (i + j) % 2
          configb[i,j] = (i + j + 1) % 2
      
    Pa = peps.create((nr,nc),pdim,configa)
    Pb = peps.create((nr,nc),pdim,configb)
    if bond == 1:
       P0 = Pa
    elif bond == 2:
       P0 = peps.add(Pa,Pb)
    P0 = peps.add_noise(P0,pdim,bond,fac=0.1)

    if rank == -1:
       PHPa = peps_hlr.eval_heish(P0, P0, auxbond, iop)
       print 'PHPa=',PHPa
       vec = peps.flatten(P0)
       energy1(vec,bond)
       print
       PHPb = spepo_hlr.eval_heish(P0, P0, iop)
       print 'PHPb=',PHPb
       exit()

    if rank == 0: print '\nStart optimization...'
    vec = peps.flatten(P0)
    # test deriv
    #print np.linalg.norm(deriv(vec)),vec[0]
    #print np.linalg.norm(deriv_mpi(vec)),vec[0]
    result = scipy.optimize.minimize(bound_energy_fn, jac=deriv_mpi, x0=vec,\
			             options={'maxiter':10})
    return 0 

if __name__ == '__main__':
   test_min()
