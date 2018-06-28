import numpy
import scipy.linalg
from zmpo_dmrg.source.mpsmpo import mpo_class 
from zmpo_dmrg.source.mpsmpo import mps_class 
import mpo_dmrg_model
from mpo_dmrg_model import cre,cret,ann,idn,sgn,nii

def getTij(n,a):
   tij = numpy.zeros((n,n))
   for i in range(n):
      tij[i,i] = 2.0
   for i in range(n-1):
      tij[i,i+1] = tij[i+1,i] = -1.0
   tij = 0.5*tij/a**2
   return tij

# h1d = -1/2*d^2/dx^2 + vne[i]
def genHmpo(n,a,vne):
   wfacs = []
   for i in range(n):
      t = 0.5/a**2
      U = 1.0/a**2+vne[i]
      tmp = mpo_dmrg_model.genHubbardSpatial(n,i,t,U)
      wfacs.append(tmp)
   tmpo = mpo_class.class_mpo(n,sites=wfacs)
   return tmpo

def genNmpo(n):
   nmpo = mpo_class.occmpo(2*n)
   nmpo = nmpo.merge([[2*i,2*i+1] for i in range(n)])
   return nmpo

def compress(mps_i1,maxM,iprt=0):
   normA = mps_i1.normalize()
   if iprt > 0:
      print '\n[compress]'
      print ' norm of exppsi before normalization =',normA
      print ' bdim0=',mps_i1.bdim()
   # Cast to canonical form
   mps_i1.leftCanon()
   if iprt > 0: print ' bdimL=',mps_i1.bdim()
   # Compress
   mps_i1.rightCanon(Dcut=maxM)
   if iprt > 0: print ' bdimR=',mps_i1.bdim()
   mps_i1.leftCanon(Dcut=maxM)
   if iprt > 0: print ' bdimL=',mps_i1.bdim()
   return normA

def randomMPS(n,D=1):
   numpy.random.seed(1)
   wfacs = [numpy.random.uniform(-1,1,size=(1,4,D))]
   for i in range(n-2): 
      wfacs.append(numpy.random.uniform(-1,1,size=(D,4,D)))
   wfacs.append(numpy.random.uniform(-1,1,size=(D,4,1)))
   mps = mps_class.class_mps(n,sites=wfacs,iop=1)
   mps.normalize()
   return mps

def prodMPS(n):
   wfacs = [numpy.ones((1,4,1))/2.0]
   for i in range(n-2): 
      wfacs.append(numpy.ones((1,4,1))/2.0)
   wfacs.append(numpy.ones((1,4,1))/2.0)
   mps = mps_class.class_mps(n,sites=wfacs,iop=1)
   print mps.dot(mps)
   return mps


def genEvenOdd(n,iop,vl,vr):
   io = numpy.zeros((1,1,4,4))
   io[0,0] = numpy.identity(4)
   if iop == 0:
      # apply t[even]
      if n%2 == 0:
         wfacs = [vl,vr]*(n/2)
         prj_mpo = mpo_class.class_mpo(n,sites=wfacs)
      else:   
         wfacs = [vl,vr]*(n/2)+[io]
         prj_mpo = mpo_class.class_mpo(n,sites=wfacs)
   else:
      # apply t[odd]
      if n%2 == 0:
         wfacs = [io]+[vl,vr]*(n/2)+[io]
         prj_mpo = mpo_class.class_mpo(n,sites=wfacs)
      else:   
         wfacs = [io]+[vl,vr]*(n/2)
         prj_mpo = mpo_class.class_mpo(n,sites=wfacs)
   return prj_mpo

# T1d = -1/2*d^2/dx^2 = [[1,-0.5,...],[-0.5,1,-0.5,...],...]/a2
def genPmpo(n,a,tau):
   # ai_a^+*aj_a + ai_b^+*aj_b # [ai]*[bj]*[ck]*[dl]=[abcd,ijkl]
   tijAA = reduce(numpy.kron,(cret,sgn,ann,idn))
   tijBB = reduce(numpy.kron,(idn,cret,sgn,ann))
   # -d^2/dx^2 [off-diagonal]
   tij = tijAA+tijBB
   tij = -0.5*(tij+tij.T) 
   e,v = scipy.linalg.eigh(tij)
   eij = numpy.einsum('ik,k,jk->ij',v,numpy.exp(-tau/a**2*e),v)
   # (abcdijkl)->(abij,cdkl)
   eij = eij.reshape((2,2,2,2,2,2,2,2))
   eij = eij.transpose(0,1,4,5,2,3,6,7)
   eij = eij.reshape((16,16))
   # not symmetric in bipartition
   u,s,vt = scipy.linalg.svd(eij)
   vl = u.reshape((4,4,16))
   vl = vl.transpose((2,0,1))
   vl = vl.reshape((1,16,4,4))
   vr = numpy.einsum('k,kj->kj',s,vt)
   vr = vr.reshape((16,1,4,4))
   # generate MPO
   prj_empo = genEvenOdd(n,0,vl,vr)
   prj_ompo = genEvenOdd(n,1,vl,vr)
   # apply t[diag]=exp(-tau/a^2*ni)
   tmp = numpy.identity(2)
   tmp[1,1] = numpy.exp(-tau/a**2)
   tmp = tmp.reshape((1,1,2,2))
   wfacs = [tmp]*(2*n)
   prj_dmpo = mpo_class.class_mpo(2*n,sites=wfacs)
   prj_dmpo = prj_dmpo.merge([[2*i,2*i+1] for i in range(n)])
   return prj_empo,prj_ompo,prj_dmpo

# Chemical potential term exp(-tau*[vi*N])
def genLmpo(n,tau,vi):
   wfacs = []
   for i in range(2*n):
      tmp = numpy.identity(2)
      tmp[1,1] = numpy.exp(-tau*vi[i])
      tmp = tmp.reshape((1,1,2,2))
      wfacs.append(tmp)
   lmpo = mpo_class.class_mpo(2*n,sites=wfacs)
   lmpo = lmpo.merge([[2*i,2*i+1] for i in range(n)])
   return lmpo

# single partilce MPS, much like sum ni*vi for MPO
def genSmps(v):
   n = len(v)
   wfacs = []
   tmp = numpy.zeros((1,4,2))
   tmp[0,0,0] = 1.0
   tmp[0,1,1] = v[0]
   wfacs.append(tmp)
   for i in range(1,n-1):
      tmp = numpy.zeros((2,4,2))
      tmp[0,0,0] = 1.0
      tmp[1,0,1] = 1.0
      tmp[0,1,1] = v[i]
      wfacs.append(tmp)
   tmp = numpy.zeros((2,4,1))
   tmp[0,1,0] = v[n-1]
   tmp[1,0,0] = 1.0
   wfacs.append(tmp)
   mps = mps_class.class_mps(n,sites=wfacs,iop=1)
   return mps 
