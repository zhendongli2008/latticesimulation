#
# Compute scale,zpeps,local2,local1a,local1b for 2D case
#
import numpy
import scipy.linalg
import genSite
from latticesimulation.ls2d import contraction2d

def initialization(n,mass2=1.0,iprt=0,auxbond=20,guess=None):
   print '\n[gen2d.initialization] n=',n,' mass2=',mass2
   # Interior
   d = 4 
   lam = 4.0+mass2
   tint = genSite.genZSite2D(lam,0)
   zpeps = numpy.empty((n,n), dtype=numpy.object)
   # Corners
   # Compute scaling factor
   scale,z = contraction2d.binarySearch(zpeps,auxbond,iprt=iprt,guess=guess)
   # Local terms
   local2  = scale*genSite.genZSite2D(lam,1)
   local1a = scale*genSite.genZSite2D(lam,2)
   local1b = scale*genSite.genZSite2D(lam,3)
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
