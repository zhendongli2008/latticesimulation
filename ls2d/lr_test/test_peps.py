import autograd.numpy as np
import autograd
import scipy.optimize
import peps_hlr
from latticesimulation.ls2d.opt_simple import peps
from latticesimulation.ls2d.opt_simple import peps_h

def test_min():
    np.random.seed(0)
    nr = 4
    nc = 4
    pdim = 2
    bond = 3
    auxbond = 6

    # interface to autograd:
    def energy1(vec):
        P = peps.aspeps(vec, (nr,nc), pdim, bond)
        PHP = peps_hlr.eval_heish(P, P, auxbond)
        PP = peps.dot(P,P,auxbond)
        e = PHP/PP
	print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
        return e 
    # dlog<P|1+tH+t^2+...|P>/dt|(t=0) = Energy
    def energy2(vec):
       def fun(x):
          P = peps.aspeps(vec, (nr,nc), pdim, bond)
          return peps_h.product(P, P, auxbond, x)
       dfun = autograd.grad(fun)
       return dfun(0.0)

    bound_energy_fn = energy1
    deriv = autograd.grad(bound_energy_fn)

    ifload = True #False 
    if not ifload:

       # Initialization
       configa = np.zeros([nr,nc], dtype=np.int)
       configb = np.zeros([nr,nc], dtype=np.int)
       for i in range(nr):
           for j in range(nc):
               configa[i,j] = (i + j) % 2
               configb[i,j] = (i + j + 1) % 2
       assert np.sum(configa)%2 == 0
       assert np.sum(configb)%2 == 0
       # initial guess by AFM
       pepsa = peps.create((nr,nc),pdim,configa)
       pepsb = peps.create((nr,nc),pdim,configb)
       peps0 = peps.add(pepsa,pepsb) # this has bond=2
       #pepsc = peps.random(peps0.shape, pdim, 1, 1.0) 
       #peps0 = peps.add(peps0, pepsc)
       peps0 = peps.add_noise(peps0,pdim,bond,fac=0.01)
       vec = peps.flatten(peps0)

       # test
       print 'nparams=',len(vec)
       print 'test energy' 
       print bound_energy_fn(vec)
       print 'test grad' 
       d = deriv(vec)

    else:

       print 'load guess...'
       vec = np.load('peps_vec2.npy')
       D0 = 2
       peps_v = peps.aspeps(vec, (nr,nc), pdim, D0)
       PP = peps.dot(peps_v,peps_v,auxbond)
       vec = vec*np.power(PP,-0.5/(nr*nc))
       # Add noise
       if bond>D0:
          peps_v = peps.aspeps(vec, (nr,nc), pdim, D0) 
	  peps_c = peps.random(peps_v.shape, pdim, bond-D0, 1.0)
          peps_t = peps.add(peps_v, peps_c)
	  print '<v|v>=',peps.dot(peps_v,peps_v,auxbond)
	  print '<v|c>=',peps.dot(peps_v,peps_c,auxbond)
	  print '<c|c>=',peps.dot(peps_c,peps_c,auxbond)
	  print '<t|t>=',peps.dot(peps_t,peps_t,auxbond)
          PP = peps.dot(peps_t,peps_t,auxbond)
          vec = peps.flatten(peps_t)*np.power(PP,-0.5/(nr*nc))
       energy = bound_energy_fn(vec)/(nr*nc)
       print 'PP =',PP,' eav =',energy

    def save_vec(vec):
	print '\ncallback...'
        fname = 'peps_vec1'
        energy = bound_energy_fn(vec)/(nr*nc)
        peps_v = peps.aspeps(vec, (nr,nc), pdim, bond)
        PP = peps.dot(peps_v,peps_v,auxbond)
        nvec = vec*np.power(PP,-0.5/(nr*nc))
        np.save(fname,nvec)
        print ' --- save vec into fname=',fname,' eav=',energy
	print
        return nvec

    # optimize
    print '\nStart optimization...'
    method = 'BFGS' #'L-BFGS-B'
    for i in range(3):
       print 'i=',i,np.linalg.norm(deriv(vec))
       vec = vec - 0.1*deriv(vec)

    for i in range(5):
       result = scipy.optimize.minimize(bound_energy_fn, jac=deriv, x0=vec,\
			                method=method,options={'maxiter':2})
       vec = save_vec(vec)
    
    P0 = peps.aspeps(result.x, (nr,nc), pdim, bond)
    print "final eav =",bound_energy_fn(peps.flatten(P0))/(nr*nc)
    return 0 


if __name__ == '__main__':
   test_min()
