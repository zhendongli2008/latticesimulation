import numpy
import gen2d
from latticesimulation.ls2d import contraction2d

# sum_{i<j} Vij*ni*nj; note that on site term is to be added in a separated PEPO.
def genNPEPO(n=6,mass2=1.0,iprt=0,auxbond=20,\
	     nij=[],psites=[],iflocal=False):
   print '\n[genPEPO.genNPEPO] n=',n,' psites=',psites
   # Generate local field sites
   scale,zpeps,local2,local1a,local1b = gen2d.initialization(n,mass2,iprt,auxbond)
   # Prepare operators
   idn = numpy.array([[1.,0.],[0.,1.]])
   if len(nij) == 0:
      ni = numpy.array([[0.,0.],[0.,1.]])
      nj = numpy.array([[0.,0.],[0.,1.]])
   else:
      ni,nj = nij
   if len(psites) == 0: psites = range(n)
   npepo = numpy.empty((n,n),dtype=numpy.object)
   np = 2
   nb = 4
   for i in range(n):
      for j in range(n):
	 # 2 - 7 - 3
	 # |   |   |
	 # 5 - 8 - 6
	 # |   |   |
	 # 0 - 4 - 1
	 l,u,d,r = zpeps[i,j].shape
	 tensor0 = numpy.einsum('pq,ludr->pqludr',idn,zpeps[i,j])
	 # 1. Determine the shapes
	 # Corners: 
	 if i==0 and j==0:
            tmp = numpy.zeros((1,nb,1,nb,np,np,l,u,d,r))
	 elif i==0 and j==n-1:
            tmp = numpy.zeros((nb,nb,1,1,np,np,l,u,d,r))
  	 elif i==n-1 and j==0:
            tmp = numpy.zeros((1,1,nb,nb,np,np,l,u,d,r))
	 elif i==n-1 and j==n-1:
            tmp = numpy.zeros((nb,1,nb,1,np,np,l,u,d,r))
	    tmp[0,0,1,0] = tensor0.copy()
	 # Edges:
	 elif i==0 and (j>0 and j<n-1):
            tmp = numpy.zeros((nb,nb,1,nb,np,np,l,u,d,r))
 	 elif (i>0 and i<n-1) and j==0:
            tmp = numpy.zeros((1,nb,nb,nb,np,np,l,u,d,r))
	 elif (i>0 and i<n-1) and j==n-1:
            tmp = numpy.zeros((nb,nb,nb,1,np,np,l,u,d,r))
	    tmp[1,1,0,0] = tensor0.copy()
	    tmp[0,1,1,0] = tensor0.copy()
         elif i==n-1 and (j>0 and j<n-1):
            tmp = numpy.zeros((nb,1,nb,nb,np,np,l,u,d,r))
	    tmp[0,0,1,0] = tensor0.copy()
	 # Interior:
	 elif (i>0 and i<n-1) and (j>0 and j<n-1):
            tmp = numpy.zeros((nb,nb,nb,nb,np,np,l,u,d,r))
	 # 2. Determine values
	 if i != n-1 or j != n-1: tmp[0,0,0,0] = tensor0.copy()
	 # For simplicity, we assume boundary do not have physical operators! 
	 if (i>0 and i<n-1) and (j>0 and j<n-1):
	    # Only i,j in psites
	    if i in psites and j in psites: 
	       tensor1i = numpy.einsum('pq,ludr->pqludr',ni,local1a)
	       tensor1j = numpy.einsum('pq,ludr->pqludr',nj,local1b)
            else:
	       tensor1i = numpy.zeros_like(tensor0)
	       tensor1j = numpy.zeros_like(tensor0)
	    if iflocal:
	       tmp[0,1,0,1] = numpy.einsum('pq,ludr->pqludr',ni,local2) 
	    # Case-4: right
	    pvec = numpy.array([1,-1,-1,1])
	    tmp[0,1,0,2] = numpy.einsum('pqludr,u->pqludr',tensor1i,pvec)
	    tmp[2,1,0,2] = numpy.einsum('pqludr,u->pqludr',tensor0,pvec)
	    tmp[2,1,0,1] = tensor1j.copy()
	    tmp[1,1,0,1] = tensor0.copy()
	    tmp[0,1,1,0] = tensor0.copy()
	    # Case-2: above
	    tmp[0,2,2,0] = numpy.einsum('pqludr,l->pqludr',tensor0,pvec)
	    tmp[0,2,0,0] = tensor1i.copy()
	    tmp[0,1,2,1] = numpy.einsum('pqludr,l->pqludr',tensor1j,pvec)
	    # Case-3: upper-right
	    tmp[0,1,2,2] = numpy.einsum('pqludr,l,u->pqludr',tensor0,pvec,pvec)
	    # Case-1: upper-left
	    tmp[0,1,0,3] = -numpy.einsum('pqludr,u->pqludr',tensor1j,pvec)
	    tmp[3,1,0,3] = numpy.einsum('pqludr,u->pqludr',tensor0,pvec)
	    tmp[3,1,2,1] = numpy.einsum('pqludr,l->pqludr',tensor0,pvec)
         # 3. Save
 	 tmp = tmp.transpose(4,5,0,6,1,7,2,8,3,9) # wxyzpqludr->pq,wl,xu,yd,zr
	 s = tmp.shape
	 npepo[i,j] = tmp.reshape(s[0],s[1],s[2]*s[3],s[4]*s[5],s[6]*s[7],s[8]*s[9]).copy()
   return npepo

def ceval(npepo,conf1,conf2,auxbond=20):
   epeps = numpy.empty(npepo.shape,dtype=numpy.object)
   # Set up <PEPSij|PEPO|PEPSij>
   for i in range(epeps.shape[0]):
      for j in range(epeps.shape[1]):
	 epeps[i,j] = npepo[i,j][conf1[i,j],conf2[i,j]]
   cab = contraction2d.contract(epeps,auxbond)
   return cab

# Test
def pepo2cpeps(npepo,palst,pblst,auxbond=20):
   na = len(palst)
   nb = len(pblst)
   cab = numpy.zeros((na,nb))
   for ia in range(na):
      for ib in range(nb):
  	 pa = palst[ia]
	 pb = pblst[ib]
   	 vac = numpy.zeros(npepo.shape,dtype=numpy.int)
	 vac[pa] = 1
	 vac[pb] = 1
	 cab[ia,ib] = ceval(npepo,vac,vac,auxbond)
   return cab


if __name__ == '__main__':
   m = 3
   n = 2*m+1
   mass2 = 1.0
   iprt = 1
   abond = 40
   iflocal = False #True
   npepo = genNPEPO(n,mass2,iprt=0,auxbond=abond,iflocal=iflocal)
   vac = numpy.zeros(npepo.shape,dtype=numpy.int)
   print '\n<0|O|0>=',ceval(npepo,vac,vac,auxbond=abond)
   
   from latticesimulation.ls2d import exact2d
   import scipy.linalg
   t2d = exact2d.genT2d(n,numpy.sqrt(mass2))
   t2d = t2d.reshape((n*n,n*n))
   tinv = scipy.linalg.inv(t2d)
   tinv = tinv.reshape((n,n,n,n))

   k = 2

   print '\nCase-0: central'
   for i in range(k):
      palst = [(m-i,m-i)]
      pblst = [(m-i,m-i)]
      # Comparison   
      cab0 = tinv[palst[0]][pblst[0]]
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0,0]

   print '\nCase-4: right'
   for i in range(k):
      palst = [(m-i,m-i)]
      pblst = [(m-i,m+i+1)]
      # Comparison   
      cab0 = tinv[palst[0]][pblst[0]]
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0,0]
      if iflocal:
         cab1i = pepo2cpeps(npepo,palst,palst,auxbond=abond)
         print 'i=',i,'pa,pa=',palst,pblst,'cab1i=',cab1i[0,0]
         cab1j = pepo2cpeps(npepo,pblst,pblst,auxbond=abond)
         print 'i=',i,'pb,pb=',pblst,pblst,'cab1j=',cab1j[0,0]
         print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0,0]
	 print 'net vij=',(cab1-cab1i-cab1j)[0,0]
      #>>> 
      palst = [(m-i,m+i+1)]
      pblst = [(m-i,m-i)]
      # Comparison   
      cab0 = tinv[palst[0]][pblst[0]]
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0,0]

   print '\nCase-3: upper-right'
   for i in range(k):
      palst = [(m-i,m-i)]
      pblst = [(m+i+1,m+i+1)]
      # Comparison   
      cab0 = tinv[palst[0]][pblst[0]]
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0,0]
      #>>>
      palst = [(m+i+1,m+i+1)]
      pblst = [(m-i,m-i)]
      # Comparison   
      cab0 = tinv[palst[0]][pblst[0]]
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0,0]

   print '\nCase-2: above'
   for i in range(k):
      palst = [(m-i,m-i)]
      pblst = [(m+i+1,m-i)]
      # Comparison   
      cab0 = tinv[palst[0]][pblst[0]]
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0,0]
      #>>>
      palst = [(m+i+1,m-i)]
      pblst = [(m-i,m-i)]
      # Comparison   
      cab0 = tinv[palst[0]][pblst[0]]
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0,0]

   print '\nCase-1: upper-left'
   for i in range(k):
      palst = [(m,m)]
      pblst = [(m+i+1,m-i-1)]
      # Comparison   
      cab0 = tinv[palst[0]][pblst[0]]
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0,0]
      #>>>
      palst = [(m+i+1,m-i-1)]
      pblst = [(m,m)]
      # Comparison   
      cab0 = tinv[palst[0]][pblst[0]]
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0,0]
