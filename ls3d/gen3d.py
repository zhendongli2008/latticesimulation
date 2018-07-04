#
# Compute scale,zpeps,local2,local1a,local1b for 2D case
#
import numpy
import scipy.linalg
import genSite
from latticesimulation.ls2d import contraction2d

# C1-E1-C2
# |  |  |
# E2-I0-E3
# |  |  | 
# C3-E4-C4
def gen_c1(info):
   return numpy.ones((1,1,1,1))
def gen_c2(info):
   return numpy.ones((1,1,1,1))
def gen_c3(info):
   return numpy.ones((1,1,1,1))
def gen_c4(info):
   return numpy.ones((1,1,1,1))
def gen_e1(info):
   return numpy.ones((1,1,1,1))
def gen_e2(info):
   return numpy.ones((1,1,1,1))
def gen_e3(info):
   return numpy.ones((1,1,1,1))
def gen_e4(info):
   return numpy.ones((1,1,1,1))
def gen_i0(info):
   return numpy.ones((1,1,1,1))

def initialization(n,mass2=1.0,iprt=0,auxbond=20,guess=None):
   print '\n[gen2d.initialization] n=',n,' mass2=',mass2
   # Interior
   n2 = n**2
   lam = 6.0+mass2
   tint = genSite.genZSite3D(lam,0)
   zpeps = numpy.empty((n2,n2), dtype=numpy.object)
   # Prepare
   info = [n,tint]
   c1 = gen_c1(info)
   c2 = gen_c2(info)
   c3 = gen_c3(info)
   c4 = gen_c4(info)
   e1 = gen_e1(info)
   e2 = gen_e2(info)
   e3 = gen_e3(info)
   e4 = gen_e4(info)
   i0 = gen_i0(info)
   # Corners
   zpeps[:n,:n] = c1.copy()
   zpeps[:n,n2-n:n2] = c2.copy()
   zpeps[n2-n:n2,:n] = c3.copy()
   zpeps[n2-n:n2,n2-n:n2] = c4.copy()
   # Edges
   for j in range(1,n-1):
      zpeps[:n,j*n:(j+1)*n] = e1.copy()
   for j in range(1,n-1):
      zpeps[n2-n:n2,j*n:(j+1)*n] = e4.copy()
   for i in range(1,n-1):
      zpeps[i*n:(i+1)*n,:n] = e2.copy()
   for i in range(1,n-1):
      zpeps[i*n:(i+1)*n,n2-n:n2] = e3.copy()
   # Inner
   for i in range(1,n-1):
      for j in range(1,n-1):
	 zpeps[i*n:(i+1)*n,j*n:(j+1)*n] = i0.copy()
   # Compute scaling factor
   scale,z = contraction2d.binarySearch(zpeps,auxbond,iprt=iprt,guess=guess)
   # Local terms
   local2  = scale*genSite.genZSite3D(lam,1)
   local1a = scale*genSite.genZSite3D(lam,2)
   local1b = scale*genSite.genZSite3D(lam,3)
   zpeps = scale*zpeps
   return scale,zpeps,local2,local1a,local1b


if __name__ == '__main__':
   m = 4
   n = 2*m+1
   mass2 = 0.2
   iprt = 1
   auxbond = 40

   result = initialization(n,mass2,iprt,auxbond)
   scale,zpeps,local2,local1a,local1b = result

#   # Test Z:
#   from latticesimulation.ls2d import exact2d
#   t2d = exact2d.genT2d(n,numpy.sqrt(mass2))
#   t2d = t2d.reshape((n*n,n*n))
#   tinv = scipy.linalg.inv(t2d)
#   tinv = tinv.reshape((n,n,n,n))
#   print '\nTest Z:'
#   print 'direct detM=',scipy.linalg.det(t2d)
#   print 'scale=',scale,'Z=',numpy.power(1.0/scale,n*n)
#
#   print '\nTest local2:'
#   v2a = tinv[0,0,0,0]
#   v2b = tinv[0,1,0,1]
#   v2c = tinv[m,m,m,m]
#   print 'direct point=',(0,0),v2a 
#   print 'direct point=',(0,1),v2b
#   print 'direct point=',(m,m),v2c
#
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[0,0] = local2[:1,:,:1,:]
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(0,0),'auxbond=',auxbond,val,val-v2a
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[0,1] = local2[:,:,:1,:]
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(0,1),'auxbond=',auxbond,val,val-v2b
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[m,m] = local2[:,:,:,:]
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(m,m),'auxbond=',auxbond,val,val-v2c
#   
#   print '\nTest local1:'
#   v1a = tinv[3,3,m-1,m-2]
#   v1b = tinv[m,m,m+3,m-2]
#   v1c = tinv[m,m,m+2,m]
#   v1d = tinv[m,m,m,m+2]
#   print 'direct point=',(3,3),(m-1,m-2),v1a
#   print 'direct point=',(m,m),(m+3,m-2),v1b
#   print 'direct point=',(m,m),(m+2,m)  ,v1c
#   print 'direct point=',(m,m),(m,m+2)  ,v1d
#
#   print '\ncase-d: right'
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[m,m] = local1a
#      epeps[m,m+2] = local1b
#      for j in range(m):
#	 epeps[m,j] = numpy.einsum('ludr,u->ludr',epeps[m,j],[1,-1,-1,1])
#      for j in range(m+2):
#	 epeps[m,j] = numpy.einsum('ludr,u->ludr',epeps[m,j],[1,-1,-1,1])
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(m,m),(m,m+2),'auxbond=',auxbond,val,val-v1d
#   # PATH
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[m,m] = local1a
#      epeps[m,m+2] = local1b
#      for j in range(m,m+2):
#	 epeps[m,j] = numpy.einsum('ludr,u->ludr',epeps[m,j],[1,-1,-1,1])
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(m,m),(m,m+2),'auxbond=',auxbond,val,val-v1d
#
#   print '\ncase-c: above'
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[m,m] = local1a
#      epeps[m+2,m] = local1b
#      for j in range(m):
#	 epeps[m,j] = numpy.einsum('ludr,u->ludr',epeps[m,j],[1,-1,-1,1])
#      for j in range(m):
#	 epeps[m+2,j] = numpy.einsum('ludr,u->ludr',epeps[m+2,j],[1,-1,-1,1])
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(m,m),(m+2,m),'auxbond=',auxbond,val,val-v1c
#   # PATH
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[m,m] = local1a
#      epeps[m+2,m] = local1b
#      for i in range(m+2,m,-1):
#	 epeps[i,m] = numpy.einsum('ludr,l->ludr',epeps[i,m],[1,-1,-1,1])
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(m,m),(m+2,m),'auxbond=',auxbond,val,val-v1c
#
#   print '\ncase-a: upper-right'
#   # Original
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[3,3] = local1a
#      epeps[m-1,m-2] = local1b
#      assert m-2>3
#      for j in range(3):
#         epeps[3,j] = numpy.einsum('ludr,u->ludr',epeps[3,j],[1,-1,-1,1])
#      for j in range(m-2):
#	 epeps[m-1,j] = numpy.einsum('ludr,u->ludr',epeps[m-1,j],[1,-1,-1,1])
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(3,3),(m-1,m-2),'auxbond=',auxbond,val,val-v1a
#   # PATH
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[3,3] = local1a
#      epeps[m-1,m-2] = local1b
#      # Changes col
#      for j in range(3,m-2):
#	 epeps[m-1,j] = numpy.einsum('ludr,u->ludr',epeps[m-1,j],[1,-1,-1,1])
#      # Changes row
#      for i in range(m-1,3,-1):
#	 epeps[i,3] = numpy.einsum('ludr,l->ludr',epeps[i,3],[1,-1,-1,1])
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(3,3),(m-1,m-2),'auxbond=',auxbond,val,val-v1a
#
#   print '\ncase-b: upper-left'
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[m,m] = local1a
#      epeps[m+3,m-2] = local1b
#      for j in range(m):
#         epeps[m,j] = numpy.einsum('ludr,u->ludr',epeps[m,j],[1,-1,-1,1])
#      for j in range(m-2):
#	 epeps[m+3,j] = numpy.einsum('ludr,u->ludr',epeps[m+3,j],[1,-1,-1,1])
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(m,m),(m+3,m-2),'auxbond=',auxbond,val,val-v1b
#   # PATH
#   for auxbond in [10,25,40]:
#      epeps = zpeps.copy()
#      epeps[m,m] = local1a
#      epeps[m+3,m-2] = -local1b
#      # Changes col
#      for j in range(m-2,m):
#	 epeps[m+3,j] = numpy.einsum('ludr,u->ludr',epeps[m+3,j],[1,-1,-1,1])
#      # Changes row
#      for i in range(m+3,m,-1):
#	 epeps[i,m] = numpy.einsum('ludr,l->ludr',epeps[i,m],[1,-1,-1,1])
#      val = contraction2d.contract(epeps,auxbond)
#      print 'point=',(m,m),(m+3,m-2),'auxbond=',auxbond,val,val-v1b
