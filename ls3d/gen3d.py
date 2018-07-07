#
# Compute scale,zpeps,local2,local1a,local1b for 2D case
#
import numpy
import scipy.linalg
import genSite
from latticesimulation.ls2d import contraction2d
import h5py

# Six directions for boundary
#    u t
#     \|
# l ---*--- r
#      |\
#      b d
l = 1; r = 1; u = 1; d = 1; t=1; b = 1

# Generic block structure (reference is I0)
def gen_block(n,tint):
   zblock = numpy.empty((n,n), dtype=numpy.object)
   iden = numpy.identity(4)
   swap = numpy.einsum('i,j->ij',[0,1,1,0],[0,1,1,0])
   swap = (-1.0)**swap
   Tswp = numpy.einsum('lr,ud,ul->ludr',iden,iden,swap)
   # lower half
   for i in range(1,n):
      for j in range(0,i):
         zblock[i,j] = Tswp
   # upper+1 half 	
   for i in range(n-1):
      for j in range(i+2,n):
         zblock[i,j] = Tswp
   # diagonal
   tmp = tint.transpose(0,1,5,4,3,2) # ludrbt->l,(u,t),b,(r,d)
   for i in range(n):
      zblock[i,i] = tmp.copy()
   # upper diagonal
   tmp = numpy.einsum('xy,ludr->lxuydr',iden,Tswp)
   for i in range(n-1):
      zblock[i,i+1] = tmp.copy()
   # reference graph taken as I0
   zblock[0,0] = zblock[0,0][:,:u,:,:,:,:].copy()
   zblock[n-1,n-1] = zblock[n-1,n-1][:,:,:,:,:,:d].copy()
   return zblock

# Merge rank-6 into rank-4 tensors
def merging(zmat0):
   zmat = zmat0.copy()
   n = zmat.shape[0]
   # diagonal
   for i in range(n):
      s = zmat[i,i].shape
      tmp = zmat[i,i].reshape(s[0],s[1]*s[2],s[3],s[4]*s[5])
      zmat[i,i] = tmp.copy()
   # upper-diagonal
   for i in range(n-1):
      s = zmat[i,i+1].shape
      tmp = zmat[i,i+1].reshape(s[0]*s[1],s[2],s[3]*s[4],s[5])
      zmat[i,i+1] = tmp.copy()
   return zmat

# Treat special parts in each case
#
# C1-E1-C2
# |  |  |
# E2-I0-E3
# |  |  | 
# C3-E4-C4
def gen_i0(zblock):
   zmat = zblock.copy()
   return merging(zmat)

def gen_e1(zblock):
   zmat = zblock.copy()
   n = zblock.shape[0]
   # diagonal
   for i in range(n):
      zmat[i,i] = zmat[i,i][:,:,:t,:,:,:].copy()
   # upper-diagonal
   for i in range(n-1):
      zmat[i,i+1] = zmat[i,i+1][:,:,:u,:,:d,:].copy()
   # upper+1 half 	
   for i in range(n-1):
      for j in range(i+2,n):
         zmat[i,j] = zmat[i,j][:,:u,:d,:].copy()
   return merging(zmat)

def gen_e3(zblock):
   zmat = zblock.copy()
   n = zblock.shape[0]
   # diagonal 
   for i in range(n):
      zmat[i,i] = zmat[i,i][:,:,:,:,:r,:].copy()
   # upper-diagonal
   for i in range(n-1):
      zmat[i,i+1] = zmat[i,i+1][:l,:,:,:,:,:r].copy()
   # upper+1 half 	
   for i in range(n-1):
      for j in range(i+2,n):
	 zmat[i,j] = zmat[i,j][:l,:,:,:r].copy()
   return merging(zmat)

def gen_e2(zblock):
   zmat = zblock.copy()
   n = zblock.shape[0]
   # diagonal
   for i in range(n):
      zmat[i,i] = zmat[i,i][:l,:,:,:,:,:].copy()
   # lower half   
   for i in range(1,n):
      for j in range(0,i):
         zmat[i,j] = zmat[i,j][:l,:,:,:r].copy()
   return merging(zmat)

def gen_e4(zblock):
   zmat = zblock.copy()
   n = zblock.shape[0]
   # diagonal
   for i in range(n):
      zmat[i,i] = zmat[i,i][:,:,:,:b,:,:].copy()
   # lower half   
   for i in range(1,n):
      for j in range(0,i):
         zmat[i,j] = zmat[i,j][:,:u,:d,:].copy()
   return merging(zmat)

# e1+e2
def gen_c1(zblock):
   zmat = zblock.copy()
   n = zblock.shape[0]
   # diagonal
   for i in range(n):
      zmat[i,i] = zmat[i,i][:l,:,:t,:,:,:].copy()
   # upper-diagonal
   for i in range(n-1):
      zmat[i,i+1] = zmat[i,i+1][:,:,:u,:,:d,:].copy()
   # upper+1 half 	
   for i in range(n-1):
      for j in range(i+2,n):
         zmat[i,j] = zmat[i,j][:,:u,:d,:].copy()
   # lower half   
   for i in range(1,n):
      for j in range(0,i):
         zmat[i,j] = zmat[i,j][:l,:,:,:r].copy()
   return merging(zmat)

# e3+e4
def gen_c4(zblock):
   zmat = zblock.copy()
   n = zblock.shape[0]
   # diagonal 
   for i in range(n):
      zmat[i,i] = zmat[i,i][:,:,:,:b,:r,:].copy()
   # upper-diagonal
   for i in range(n-1):
      zmat[i,i+1] = zmat[i,i+1][:l,:,:,:,:,:r].copy()
   # upper+1 half 	
   for i in range(n-1):
      for j in range(i+2,n):
	 zmat[i,j] = zmat[i,j][:l,:,:,:r].copy()
   # lower half   
   for i in range(1,n):
      for j in range(0,i):
         zmat[i,j] = zmat[i,j][:,:u,:d,:].copy()
   return merging(zmat)

# similar to e3
def gen_c2(zblock):
   zmat = zblock.copy()
   n = zblock.shape[0]
   # diagonal 
   for i in range(n):
      zmat[i,i] = zmat[i,i][:,:,:t,:,:r,:].copy()
   # upper-diagonal
   for i in range(n-1):
      zmat[i,i+1] = zmat[i,i+1][:l,:,:u,:,:d,:r].copy()
   # upper+1 half 	
   for i in range(n-1):
      for j in range(i+2,n):
	 zmat[i,j] = numpy.ones(1).reshape(1,1,1,1)
   return merging(zmat)

# similar to e4
def gen_c3(zblock):
   zmat = zblock.copy()
   n = zblock.shape[0]
   # diagonal
   for i in range(n):
      zmat[i,i] = zmat[i,i][:l,:,:,:b,:,:].copy()
   # lower half   
   for i in range(1,n):
      for j in range(0,i):
         zmat[i,j] = numpy.ones(1).reshape(1,1,1,1)
   return merging(zmat)

def initialization(n,mass2=1.0,iprt=0,auxbond=20,guess=None):
   print '\n[gen3d.initialization] n=',n,' mass2=',mass2
   # Interior
   n2 = n**2
   lam = 6.0+mass2
   tint = genSite.genZSite3D(lam,0)
   zpeps = numpy.empty((n2,n2), dtype=numpy.object)
   # Prepare
   zblock = gen_block(n,tint)
   i0 = gen_i0(zblock)
   c1 = gen_c1(zblock)
   c2 = gen_c2(zblock)
   c3 = gen_c3(zblock)
   c4 = gen_c4(zblock)
   e1 = gen_e1(zblock)
   e2 = gen_e2(zblock)
   e3 = gen_e3(zblock)
   e4 = gen_e4(zblock)
   # Note that C3 is (0,0) 
   # C1-E1-C2
   # |  |  |
   # E2-I0-E3
   # |  |  | 
   # C3-E4-C4
   # New ordering:
   # c3-e4-c4
   for i in range(n):
      for j in range(n):
         zpeps[i,j] = c3[n-1-i,j].copy()
   for i in range(n):
     for k in range(1,n-1): 
       for j in range(n):
         zpeps[i,j+k*n] = e4[n-1-i,j].copy()
   for i in range(n):
      for j in range(n):
         zpeps[i,j+n2-n] = c4[n-1-i,j].copy()
   # e2-i0-e3
   for j in range(n):
     for k in range(1,n-1): 
       for i in range(n):
         zpeps[i+k*n,j] = e2[n-1-i,j].copy()
   for l in range(1,n-1): 
    for j in range(n):
      for k in range(1,n-1): 
        for i in range(n):
         zpeps[i+k*n,j+l*n] = i0[n-1-i,j].copy()
   for j in range(n):
     for k in range(1,n-1): 
       for i in range(n):
         zpeps[i+k*n,j+n2-n] = e3[n-1-i,j].copy()
   # c1-e1-c2
   for i in range(n):
      for j in range(n):
         zpeps[i+n2-n,j] = c1[n-1-i,j].copy()
   for i in range(n):
     for k in range(1,n-1): 
       for j in range(n):
         zpeps[i+n2-n,j+k*n] = e1[n-1-i,j].copy()
   for i in range(n):
      for j in range(n):
         zpeps[i+n2-n,j+n2-n] = c2[n-1-i,j].copy()
   # Compute scaling factor
   scale,z = contraction2d.binarySearch(zpeps,auxbond,iprt=iprt,guess=guess)
   print
   print 'scale=',scale,'n3d=',n,'Zphys=',1.0/scale**(n2**2)
   print
   # Local terms
   local2  = scale*genSite.genZSite3D(lam,1)
   local1a = scale*genSite.genZSite3D(lam,2)
   local1b = scale*genSite.genZSite3D(lam,3)
   # convert to preorder
   local2  = local2.transpose(0,1,5,4,3,2).copy()  # ludrbt->l,(u,t),b,(r,d)
   local1a = local1a.transpose(0,1,5,4,3,2).copy() # ludrbt->l,(u,t),b,(r,d)
   local1b = local1b.transpose(0,1,5,4,3,2).copy() # ludrbt->l,(u,t),b,(r,d)
   zpeps = scale*zpeps
   return scale,zpeps,local2,local1a,local1b

def tensor_dump(scale,zpeps,local2,local1a,local1b,fname='tensor'):
   print '\n[gen3d.tensor_dump] fname=',fname
   f = h5py.File(fname+'.h5','w')
   n2 = zpeps.shape[0]
   f['n2'] = n2
   f['scale'] = scale
   for i in range(n2): 
      for j in range(n2): 
         f['zpeps_'+str(i)+'_'+str(j)] = zpeps[i,j]
   f['local2'] = local2
   f['local1a'] = local1a
   f['local1b'] = local1b
   f.close()
   return 0

def tensor_load(fname='tensor'):
   print '\n[gen3d.tensor_load] fname=',fname
   f = h5py.File(fname+'.h5','r')
   n2 = f['n2'].value
   scale = f['scale'].value
   n = int(numpy.sqrt(n2))
   print ' n,n2=',(n,n2)
   print ' scale=',scale,'n3d=',n,'Zphys=',1.0/scale**(n2**2)
   zpeps = numpy.empty((n2,n2), dtype=numpy.object)
   for i in range(n2):
      for j in range(n2):
         zpeps[i,j] = f['zpeps_'+str(i)+'_'+str(j)].value
   local2 = f['local2'].value
   local1a = f['local1a'].value
   local1b = f['local1b'].value
   f.close()
   return scale,zpeps,local2,local1a,local1b 

# Z-direction
def test_zdir(m,n,scale,zpeps,local2,local1a,local1b,auxbond,off=1):
   x0,y0 = m+m*n,m+m*n
   x1,y1 = m+(m+off)*n,m+m*n
   print '\n[test_zdir] A/B=',(x0,y0),(x1,y1)
   epeps = zpeps.copy()
   # c
   s = local1a.shape
   tmp = local1a.reshape(s[0],s[1]*s[2],s[3],s[4]*s[5])
   epeps[x0,y0] = tmp.copy()
   # cbar
   s = local1b.shape
   tmp = local1b.reshape(s[0],s[1]*s[2],s[3],s[4]*s[5])
   epeps[x1,y1]= tmp.copy()
   # z-path
   vp = [1,-1,-1,1]
   for i in range(x0+1,x1+1):
      tmp = epeps[i,y0].copy()
      s = tmp.shape
      if s[0] == 4:
         epeps[i,y0] = numpy.einsum('ludr,l->ludr',tmp,vp)
      elif s[0] == 16:
         sp = numpy.einsum('i,j->ij',vp,vp)
         sp = sp.reshape(16)
	 epeps[i,y0] = numpy.einsum('ludr,l->ludr',tmp,sp)
      else:
         print 'error'
	 exit()
      print ' zcoord=',(i,y0),' bond=',s[0]
   val = contraction2d.contract(epeps,auxbond)
   return val

# D-direction
def test_ddir(m,n,scale,zpeps,local2,local1a,local1b,auxbond,off=1):
   x0,y0 = m+m*n,m+m*n
   x1,y1 = (m+(m+off)*n,m+(m+off)*n)
   print '\n[test_ddir] A/B=',(x0,y0),(x1,y1)
   epeps = zpeps.copy()
   # c
   s = local1a.shape
   tmp = local1a.reshape(s[0],s[1]*s[2],s[3],s[4]*s[5])
   epeps[x0,y0] = tmp.copy()
   # cbar
   s = local1b.shape
   tmp = local1b.reshape(s[0],s[1]*s[2],s[3],s[4]*s[5])
   epeps[x1,y1]= tmp.copy()
   # z-path
   vp = [1,-1,-1,1]
   for i in range(x0+1,x1+1):
      tmp = epeps[i,y0].copy()
      s = tmp.shape
      if s[0] == 4:
         epeps[i,y0] = numpy.einsum('ludr,l->ludr',tmp,vp)
      elif s[0] == 16:
         sp = numpy.einsum('i,j->ij',vp,vp)
         sp = sp.reshape(16)
	 epeps[i,y0] = numpy.einsum('ludr,l->ludr',tmp,sp)
      else:
         print 'error'
	 exit()
      print ' zcoord=',(i,y0),' bond=',s[0]
   # h-path
   for j in range(y0,y1):
      tmp = epeps[x1,j].copy()
      s = tmp.shape
      if s[1] == 4:
         epeps[x1,j] = numpy.einsum('ludr,u->ludr',tmp,vp)
      elif s[1] == 16:
         sp = numpy.einsum('i,j->ij',vp,vp)
         sp = sp.reshape(16)
	 epeps[x1,j] = numpy.einsum('ludr,u->ludr',tmp,sp)
      else:
         print 'error'
	 exit()
      print ' hcoord=',(x1,j),' bond=',s[1]
   val = contraction2d.contract(epeps,auxbond)
   return val

# D-direction
def test_ddir3(m,n,scale,zpeps,local2,local1a,local1b,auxbond,off=1):
   x0,y0 = m+m*n,m+m*n
   x1,y1 = (m+off+(m+off)*n,m-off+(m+off)*n)
   print '\n[test_ddir3] A/B=',(x0,y0),(x1,y1)
   epeps = zpeps.copy()
   # c
   s = local1a.shape
   tmp = local1a.reshape(s[0],s[1]*s[2],s[3],s[4]*s[5])
   epeps[x0,y0] = tmp.copy()
   # cbar
   s = local1b.shape
   tmp = local1b.reshape(s[0],s[1]*s[2],s[3],s[4]*s[5])
   epeps[x1,y1]= tmp.copy()
   # z-path
   vp = [1,-1,-1,1]
   for i in range(x0+1,x1+1):
      tmp = epeps[i,y0].copy()
      s = tmp.shape
      if s[0] == 4:
         epeps[i,y0] = numpy.einsum('ludr,l->ludr',tmp,vp)
      elif s[0] == 16:
         sp = numpy.einsum('i,j->ij',vp,vp)
         sp = sp.reshape(16)
	 epeps[i,y0] = numpy.einsum('ludr,l->ludr',tmp,sp)
      else:
         print 'error'
	 exit()
      print ' zcoord=',(i,y0),' bond=',s[0]
   # h-path
   for j in range(y0,y1):
      tmp = epeps[x1,j].copy()
      s = tmp.shape
      if s[1] == 4:
         epeps[x1,j] = numpy.einsum('ludr,u->ludr',tmp,vp)
      elif s[1] == 16:
         sp = numpy.einsum('i,j->ij',vp,vp)
         sp = sp.reshape(16)
	 epeps[x1,j] = numpy.einsum('ludr,u->ludr',tmp,sp)
      else:
         print 'error'
	 exit()
      print ' hcoord=',(x1,j),' bond=',s[1]
   val = contraction2d.contract(epeps,auxbond)
   return val

if __name__ == '__main__':
   m = 2
   n = 2*m+1
   n2 = n**2
   mass2 = 1.0
   iop = 1
   iprt = 1
   auxbond = 50
 
   from latticesimulation.ls2d import exact2d
   mass = numpy.sqrt(mass2)
   t3d = exact2d.genT3d(n,mass)
   t3d = t3d.reshape((n**3,n**3))
   det = scipy.linalg.det(t3d)
   tinv = scipy.linalg.inv(t3d)
   tinv = tinv.reshape((n,n,n,n,n,n))
   print '\nTest Z with mass =',mass
   print 'direct detM=',det,' scale0=',pow(det,-1.0/n**3)
   print tinv[m,m,m,m+1,m,m]
   print tinv[m,m,m,m,m+1,m]
   print tinv[m,m,m,m,m,m+1]
   print tinv[m,m,m,m,m+1,m+1]
   print tinv[m,m,m,m+1,m,m+1]
   print tinv[m,m,m,m+1,m+1,m]
   print tinv[m,m,m,m+1,m+1,m+1]
   print tinv[m,m,m,m-1,m-1,m-1]

   if iop == 0:
      result = initialization(n,mass2,iprt,auxbond)
      scale,zpeps,local2,local1a,local1b = result
      tensor_dump(scale,zpeps,local2,local1a,local1b)
   else:
      scale,zpeps,local2,local1a,local1b = tensor_load()
      val = contraction2d.contract(zpeps,auxbond)
      print '\npart=',val
      val = test_zdir(m,n,scale,zpeps,local2,local1a,local1b,auxbond,off=1)
      print '\nzdir=',val
      val = test_ddir(m,n,scale,zpeps,local2,local1a,local1b,auxbond,off=1)
      print '\nddir=',val
      val = test_ddir3(m,n,scale,zpeps,local2,local1a,local1b,auxbond,off=1)
      print '\nddir=',val
