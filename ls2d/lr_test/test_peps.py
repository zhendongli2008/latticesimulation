import autograd.numpy as np
import autograd
import scipy.optimize
from latticesimulation.ls2d.opt_simple import peps
from latticesimulation.ls2d.opt_simple import peps_h
import peps_hlr
import spepo_hlr

def test_min():
    np.random.seed(0)
    nr = 4
    nc = 4
    pdim = 2
    bond = 2
    auxbond = bond**2

    # interface to autograd:
    def energy1(vec, bond):
       P = peps.aspeps(vec, (nr,nc), pdim, bond)
       PHP = peps_hlr.eval_heish(P, P, auxbond)
       PP = peps.dot(P,P,auxbond)
       e = PHP/PP
       print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
       return e 
    # dlog<P|1+tH+t^2+...|P>/dt|(t=0) = Energy
    def energy2(vec, bond):
       P = peps.aspeps(vec, (nr,nc), pdim, bond)
       PHP = spepo_hlr.eval_heish(P, P)
       PP = peps.dot(P,P,auxbond)
       e = PHP/PP
       print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
       return e 

    bound_energy_fn = lambda x: energy1(x,bond)
    deriv = autograd.grad(bound_energy_fn)

    ifload = False 
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
       #pepsc = peps.random(peps0.shape, pdim, 1, 0.01) 
       #peps0 = peps.add(peps0, pepsc)
       peps0 = peps.add_noise(peps0,pdim,bond,fac=0.1)
       vec = peps.flatten(peps0)

       # test
       print 'energy2=',energy2(vec, bond)
       print 'energy1=',energy1(vec, bond)
       exit()
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
	  peps_t = peps.add_noise(peps_t,pdim,bond,fac=0.1)
	  PP = peps.dot(peps_t,peps_t,auxbond)
          vec = peps.flatten(peps_t)*np.power(PP,-0.5/(nr*nc))
	  print '<v|v>=',peps.dot(peps_v,peps_v,auxbond)
	  print '<v|c>=',peps.dot(peps_v,peps_c,auxbond)
	  print '<c|c>=',peps.dot(peps_c,peps_c,auxbond)
	  print '<t|t>=',peps.dot(peps_t,peps_t,auxbond)
	  print '<t|v>=',peps.dot(peps_t,peps_v,auxbond)
	  print '<t|c>=',peps.dot(peps_t,peps_c,auxbond)
	  print 'e_v=',energy1(peps.flatten(peps_v),2) 
	  print 'e_c=',energy1(peps.flatten(peps_c),1)
	  print 'e_t=',energy1(peps.flatten(peps_t),3)

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
    step = 1.0
    for i in range(3):
       g = deriv(vec)
       print 'i=',i,np.linalg.norm(g)
       vec = vec - step*g

    print '\nStart optimization...'
    result = scipy.optimize.minimize(bound_energy_fn, jac=deriv, x0=vec,\
			             options={'maxiter':10},callback=save_vec)
    
    P0 = peps.aspeps(result.x, (nr,nc), pdim, bond)
    print "final eav =",bound_energy_fn(peps.flatten(P0))/(nr*nc)
    return 0 


if __name__ == '__main__':
   test_min()
