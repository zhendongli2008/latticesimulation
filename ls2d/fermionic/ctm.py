import numpy
import scipy.linalg

# C1 - T1 - C2
# |    |    |
# T4 - M0 - T2
# |    |    |
# C4 - T3 - C3
def extractCTM(zpeps0):
   c4 = zpeps0[0,0].copy()
   t3 = zpeps0[0,1].copy()
   c3 = zpeps0[0,-1].copy()
   t4 = zpeps0[1,0].copy()
   m0 = zpeps0[1,1].copy()
   t2 = zpeps0[1,-1].copy()
   c1 = zpeps0[-1,0].copy()
   t1 = zpeps0[-1,1].copy()
   c2 = zpeps0[-1,-1].copy()
   sitelst = [m0,c1,c2,c3,c4,t1,t2,t3,t4]
   return sitelst

def convertPEPS(n,sitelst):
   m0,c1,c2,c3,c4,t1,t2,t3,t4 = sitelst
   zpeps = numpy.empty((n,n), dtype=numpy.object)
   for i in range(1,n-1):
      for j in range(1,n-1):
         zpeps[i,j] = m0.copy()
   # Corners
   zpeps[0,0]     = c4.copy()
   zpeps[0,n-1]   = c3.copy()
   zpeps[n-1,0]   = c1.copy()
   zpeps[n-1,n-1] = c2.copy()
   # Boundaries
   for j in range(1,n-1):
      zpeps[0,j]   = t3.copy()
      zpeps[j,0]   = t4.copy()
      zpeps[n-1,j] = t1.copy()
      zpeps[j,n-1] = t2.copy()
   return zpeps

# Left insertion + renormalization
def leftCTM(sitelst,chi):
   m0,c1,c2,c3,c4,t1,t2,t3,t4 = sitelst 
   # corner 
   c1_t1 = numpy.einsum('ludr,rUDR->luUdDR',c1,t1)
   s = c1_t1.shape
   assert s[0]==s[1]==s[2]==1
   c1_t1 = c1_t1.reshape([s[3]*s[4],s[5]]) # DR
   # center
   t4_m0 = numpy.einsum('ludr,rUDR->luUdDR',t4,m0)
   s = t4_m0.shape
   assert s[0]==1
   t4_m0 = t4_m0.reshape([s[1]*s[2],s[3]*s[4],s[5]]) # UDR
   # corner
   c4_t3 = numpy.einsum('ludr,rUDR->luUdDR',c4,t3)
   s = c4_t3.shape
   assert s[0]==s[3]==s[4]==1
   c4_t3 = c4_t3.reshape([s[1]*s[2],s[5]]) # UR
   # DM
   dm = c1_t1.dot(c1_t1.T) + c4_t3.dot(c4_t3.T)
   e,v = scipy.linalg.eigh(-dm)
   e = -e
   n = len(e)
   n = len(numpy.argwhere(e>0.0))
   print ' n=',n,' eigs=',e[:4]
   z = v[:,:min(n,chi)].copy()
   # renormalization
   c1z = numpy.einsum('DR,Dd->dR',c1_t1,z)
   c4z = numpy.einsum('UR,Uu->uR',c4_t3,z)
   t4z = numpy.einsum('UDR,Uu,Dd->udR',t4_m0,z,z)
   # update
   s = c1z.shape
   c1 = c1z.reshape([1,1,s[0],s[1]]).copy()
   s = c4z.shape
   c4 = c4z.reshape([1,s[0],1,s[1]]).copy()
   s = t4z.shape
   t4 = t4z.reshape([1,s[0],s[1],s[2]]).copy()
   # sitelst
   sitelst_new = [m0,c1,c2,c3,c4,t1,t2,t3,t4]
   return sitelst_new

def rightCTM(sitelst,chi):
   sitelst_new = []
   # LR inversion
   for site0 in sitelst:
      site1 = numpy.einsum('ludr->rudl',site0)
      sitelst_new.append(site1)
   m0,c1,c2,c3,c4,t1,t2,t3,t4 = sitelst_new
   sitelst_new = [m0,c2,c1,c4,c3,t1,t4,t3,t2] 
   # leftCTM
   sitelst_new = leftCTM(sitelst_new,chi)
   # Flip back
   m0,c2,c1,c4,c3,t1,t4,t3,t2 = sitelst_new
   sitelst_tmp = [m0,c1,c2,c3,c4,t1,t2,t3,t4]
   # Set back
   sitelst_new = []
   for site0 in sitelst_tmp:
      site1 = numpy.einsum('rudl->ludr',site0)
      sitelst_new.append(site1)
   return sitelst_new

def upCTM(sitelst,chi):
   sitelst_new = []
   # Counterclockwise rotation 90deg
   for site0 in sitelst:
      site1 = numpy.einsum('ludr->urld',site0)
      sitelst_new.append(site1)
   m0,c1,c2,c3,c4,t1,t2,t3,t4 = sitelst_new
   sitelst_new = [m0,c2,c3,c4,c1,t2,t3,t4,t1]
   # leftCTM
   sitelst_new = leftCTM(sitelst_new,chi)
   # Flip back
   m0,c2,c3,c4,c1,t2,t3,t4,t1 = sitelst_new
   sitelst_tmp = [m0,c1,c2,c3,c4,t1,t2,t3,t4]
   # Set back
   sitelst_new = []
   for site0 in sitelst_tmp:
      site1 = numpy.einsum('urld->ludr',site0)
      sitelst_new.append(site1)
   return sitelst_new

def downCTM(sitelst,chi):
   sitelst_new = []
   # Clockwise rotation 90deg
   for site0 in sitelst:
      site1 = numpy.einsum('ludr->dlru',site0)
      sitelst_new.append(site1)
   m0,c1,c2,c3,c4,t1,t2,t3,t4 = sitelst_new
   sitelst_new = [m0,c4,c1,c2,c3,t4,t1,t2,t3]
   # leftCTM
   sitelst_new = leftCTM(sitelst_new,chi)
   # Flip back
   m0,c4,c1,c2,c3,t4,t1,t2,t3 = sitelst_new
   sitelst_tmp = [m0,c1,c2,c3,c4,t1,t2,t3,t4]
   # Set back
   sitelst_new = []
   for site0 in sitelst_tmp:
      site1 = numpy.einsum('dlru->ludr',site0)
      sitelst_new.append(site1)
   return sitelst_new

def performCTM(zpeps,chi,nsteps):
   symbols = ['m0','c1','c2','c3','c4','t1','t2','t3','t4']
   sitelst = extractCTM(zpeps)
   for istep in range(nsteps):
      print 'istep of CTM=',istep,' chi=',chi
      sitelst_new = []
      for idx,site in enumerate(sitelst):
	 amax = numpy.max(numpy.abs(site))
	 site_new = site/amax
         print ' idx=',idx,symbols[idx],amax
	 sitelst_new.append(site_new)
      sitelst = leftCTM(sitelst_new,chi)
      sitelst = rightCTM(sitelst,chi)
      sitelst = upCTM(sitelst,chi)
      sitelst = downCTM(sitelst,chi)
   return sitelst

#
# Formation of PEPS
#
def formPEPS0(n,zpeps0):
   sitelst = extractCTM(zpeps0)
   zpeps = convertPEPS(n,sitelst)
   return zpeps

def formPEPS1(n,zpeps0,chi=20,nsteps=2):
   sitelst = performCTM(zpeps0,chi,nsteps)
   zpeps = convertPEPS(n,sitelst)
   return zpeps
