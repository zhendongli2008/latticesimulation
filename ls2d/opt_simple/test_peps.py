import autograd.numpy as np
import peps
import peps_h
import autograd
import scipy.optimize

def test_min():
    np.random.seed(5)
    nr = 4
    nc = 4
    pdim = 2
    bond = 2
    auxbond = 4
    # interface to autograd
    def energy_fn(vec, pdim,bond):
        P = peps.aspeps(vec, (nr,nc), pdim, bond)
        PHP = peps_h.eval_heish(P, P, auxbond)
        PP = peps.dot(P,P,auxbond)
        e = PHP/PP
	print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
        return PHP,PP,e
    def bound_energy_fn(vec):
        return energy_fn(vec, pdim, bond)[-1]
    deriv = autograd.grad(bound_energy_fn)

    # Initialization
    configa = np.zeros([nr,nc], dtype=np.int)
    configb = np.zeros([nr,nc], dtype=np.int)
    for i in range(nr):
        for j in range(nc):
            configa[i,j] = (i + j) % 2
            configb[i,j] = (i + j + 1) % 2
    assert np.sum(configa)%2 == 0
    assert np.sum(configb)%2 == 0
    pepsa = peps.create((nr,nc),pdim,configa)
    pepsb = peps.create((nr,nc),pdim,configb)

    # initial guess by AFM
    pepsc = peps.random(pepsa.shape, pdim, bond-2) 
    peps0 = peps.add(pepsa,pepsb) # this has bond=2
    peps0 = peps.add(peps0, pepsc)
    peps0 = peps.add_noise(peps0,pdim,bond,fac=1.e-1)
    vec = peps.flatten(peps0)

    # test
    print 'nparams=',len(vec)
    print 'test energy' 
    print bound_energy_fn(vec)
    print 'test grad' 
    d = deriv(vec)

    def save_vec(vec):
	print 'callback...'
        fname = 'peps_vec1'
        PHP,PP,e = energy_fn(vec, pdim,bond)
        vec[:] = vec*np.power(PP,-0.5/(nr*nc))
        np.save(fname,vec)
        print ' --- save vec into fname=',fname,' eav=',e/(nr*nc)
	print
        return 0

    ifload = False 
    if ifload:
       print 'load guess...'
       vec = np.load('peps_vec1.npy')
       energy = bound_energy_fn(vec)/(nr*nc)
       peps0 = peps.aspeps(vec, (nr,nc), pdim, bond)
       PP = peps.dot(peps0,peps0,auxbond)
       vec = vec*np.power(PP,-0.5/(nr*nc))
       energy = bound_energy_fn(vec)/(nr*nc)
       ## Increase
       #pepsc = peps.zeros(peps0.shape, pdim, bond-2) 
       #peps0 = peps.add(peps0, pepsc)
       #peps0 = peps.add_noise(peps0,pdim,bond,fac=1.e-1)
       #vec = peps.flatten(peps0)
       print 'eav =',energy

    # optimize
    print '\nStart optimization...'
    result = scipy.optimize.minimize(bound_energy_fn, jac=deriv, x0=vec,\
		    		     tol=1.e-4, callback=save_vec)
    print "max value", np.max(result.x)
    P0 = peps.aspeps(result.x, (nr,nc), pdim, bond)
    print "final eav =",bound_energy_fn(peps.flatten(P0))/(nr*nc)
    return 0 


if __name__ == '__main__':
   test_min()
