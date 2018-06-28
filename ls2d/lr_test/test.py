#import autograd.numpy as np
import numpy as np
import autograd
import scipy.optimize
from latticesimulation.ls2d.opt_simple import peps
from latticesimulation.ls2d.opt_simple import peps_h
import peps_hlr
import spepo_hlr

iop = 3

def test_min():
    np.random.seed(0)
    nr = 4
    nc = 4
    pdim = 2
    bond = 2
    auxbond = 4

    # interface to autograd:
    def energy1(vec, bond):
       P = peps.aspeps(vec, (nr,nc), pdim, bond)
       PHP = peps_hlr.eval_heish(P, P, auxbond, iop)
       PP = peps.dot(P,P,auxbond)
       e = PHP/PP
       print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
       return e 
    # dlog<P|1+tH+t^2+...|P>/dt|(t=0) = Energy
    def energy2(vec, bond):
       P = peps.aspeps(vec, (nr,nc), pdim, bond)
       PHP = spepo_hlr.eval_heish(P, P, iop)
       PP = peps.dot(P,P,auxbond)
       e = PHP/PP
       print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
       return e 

    bound_energy_fn = lambda x: energy1(x,bond)
    deriv = autograd.grad(bound_energy_fn)

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

    PHPa = peps_hlr.eval_heish(P0, P0, auxbond, iop)
    print 'PHPa=',PHPa
    vec = peps.flatten(P0)
    energy1(vec,bond)
    print
    PHPb = spepo_hlr.eval_heish(P0, P0, iop)
    print 'PHPb=',PHPb
    return 0 


if __name__ == '__main__':
   test_min()
