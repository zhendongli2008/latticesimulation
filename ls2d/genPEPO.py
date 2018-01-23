import numpy
import num2d
from isingMapping import mass2c
import contraction2d

# Input peps to pepo with mnludr
#
# iop=0 is wrong for the following conf:
#
#  0---1---0      0---1---0 0---0---0
#  |   |   |      |   |[0]| |   |   |  
#  0---1---0  =>  0---0---0 0---1---0  Mixed. 
#  |   |   |      |   |   | |   |   |
#  1---1---0      1---0---0 0---1---0  
#
# Only the use of [0] can kill the second configuration!
#
def genNPEPO(n=6,mass2=1.0,ng=2,iprt=0,auxbond=20,iop=1,\
	     nij=[],psites=[]):
   print '\n[genPEPO.genNPEPO] n=',n,' psites=',psites
   zpeps,local1,local2 = num2d.initialization(n,mass2,ng,iprt,auxbond)
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
	 # Interior
	 elif (i>0 and i<n-1) and (j>0 and j<n-1):
            tmp = numpy.zeros((nb,nb,nb,nb,np,np,l,u,d,r))
	 # 2. Determine values
	 if i != n-1 or j != n-1: tmp[0,0,0,0] = tensor0.copy()
	 # For simplicity, we assume boundary do not have physical operators! 
	 if (i>0 and i<n-1) and (j>0 and j<n-1):
	    # Only i,j in psites
	    if i in psites and j in psites: 
	       tensor1i = numpy.einsum('pq,ludr->pqludr',ni,local1)
	       tensor1j = numpy.einsum('pq,ludr->pqludr',nj,local1)
            else:
	       tensor1i = numpy.zeros_like(tensor0)
	       tensor1j = numpy.zeros_like(tensor0)
	    # Case-4: right
	    tmp[0,1,0,2] = tensor1i.copy() 
	    tmp[2,1,0,2] = tensor0.copy()
	    tmp[2,1,0,1] = tensor1j.copy()
	    tmp[1,1,0,1] = tensor0.copy()
	    tmp[0,1,1,0] = tensor0.copy()
	    # Case-2: above
	    tmp[0,2,2,0] = tensor0.copy()
	    if iop == 0:
	       tmp[0,2,0,1] = tensor1i.copy()
	       tmp[0,1,2,0] = tensor1j.copy()
	    else:
	       tmp[0,2,0,0] = tensor1i.copy()
	       tmp[0,1,2,1] = tensor1j.copy()
	    # Case-3: upper-right
	    tmp[0,1,2,2] = tensor0.copy()
	    if iop == 0:
	       tmp[2,1,1,2] = tensor0.copy()
	       tmp[2,1,1,0] = tensor1j.copy()
	    else:
       	       # The same as right
	       tmp[2,1,0,2] = tensor0.copy()
	       tmp[2,1,0,1] = tensor1j.copy()
	    # Case-1: upper-left
	    tmp[0,1,0,3] = tensor1j.copy()
	    tmp[3,1,0,3] = tensor0.copy()
	    if iop == 0:
	       tmp[3,1,2,0] = tensor0.copy()
	    else:
	       tmp[3,1,2,1] = tensor0.copy()
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
   ng = 2
   m = 2
   n = 2*m+1
   k = 1
   abond = 30
   mass2 = 0.1*mass2c
   npepo = genNPEPO(n,mass2,ng,iprt=0,auxbond=abond,iop=0)
   npepo2 = genNPEPO(n,mass2,ng,iprt=0,auxbond=abond,iop=1)
   vac = numpy.zeros(npepo.shape,dtype=numpy.int)
   print '<0|O|0>=',ceval(npepo,vac,vac,auxbond=abond)
   print '<0|O|0>=',ceval(npepo2,vac,vac,auxbond=abond)

   # Benchmark
   def benchmark(vac):
      palst = []
      for i in range(vac.shape[0]):
         for j in range(vac.shape[1]):
	    if vac[i,j] == 1: palst.append((i,j))
      pblst = palst
      val = 0.0
      for i in range(len(palst)):
         for j in range(len(pblst)):
            pa = palst[i]
            pb = pblst[j]
            if pa < pb:
      	       cab0 = num2d.correlationFunctions(n,mass2,ng,\
			  palst=[pa],pblst=[pb],iprt=0,auxbond=abond)
               val += cab0[0,0]
      return val

   def checkDetails(vac):
      print '\n[checkDetails]'
      palst = []
      for i in range(vac.shape[0]):
         for j in range(vac.shape[1]):
	    if vac[i,j] == 1: palst.append((i,j))
      import itertools
      nn = len(palst)
      subsets = []
      for i in range(nn+1):
         subsets += list(itertools.combinations(range(nn),i))
      print ' nn=',nn,'subsets=',len(subsets)
      print ' ',subsets
      idx = 0
      for selection in subsets:
   	 vac = numpy.zeros(npepo.shape,dtype=numpy.int)
	 pts = []
	 for i in selection:
	    vac[palst[i]] = 1
	    pts.append(palst[i])
	 val1 = ceval(npepo,vac,vac,auxbond=20)
	 val2 = ceval(npepo2,vac,vac,auxbond=20)
	 idx += 1
	 print ' idx=',idx,selection,pts,val1,val2,val2-val1
      return 0

   ista = [-1]*9 #[1]*9 #[0,1,0,0,1,0,1,0,1]
   for i1 in range(ista[0],-1,-1):
    for i2 in range(ista[1],-1,-1):
     for i3 in range(ista[2],-1,-1):
      for i4 in range(ista[3],-1,-1):
       for i5 in range(ista[4],-1,-1):
        for i6 in range(ista[5],-1,-1):
         for i7 in range(ista[6],-1,-1):
          for i8 in range(ista[7],-1,-1):
           for i9 in range(ista[8],-1,-1):
   	      vac = numpy.zeros(npepo.shape,dtype=numpy.int)
	      vac[(1,1)] = i1
	      vac[(1,2)] = i2
	      vac[(1,3)] = i3
	      vac[(2,1)] = i4
	      vac[(2,2)] = i5
	      vac[(2,3)] = i6
	      vac[(3,1)] = i7
	      vac[(3,2)] = i8
	      vac[(3,3)] = i9
   	      #vac = numpy.zeros(npepo.shape,dtype=numpy.int)
	      #vac[(1,2)] = 1
	      #vac[(2,2)] = 1
	      #vac[(3,2)] = 1
	      #vac[(3,1)] = 1
	      #print vac
   	      #checkDetails(vac)
   	      #exit()
	      val0 = benchmark(vac)
	      val1 = ceval(npepo,vac,vac,auxbond=20)
	      val2 = ceval(npepo2,vac,vac,auxbond=20)
	      print 'conf=',(i1,i2,i3,i4,i5,i6,i7,i8,i9),\
		    'val1=',val1,'val2=',val2,'diff1=',val1-val0,'diff2=',val2-val0

   print 'Case-0: local terms ni'
   for i in range(k):
      palst = [(m+i,m+i)]
      # Comparison   
      cab0 = num2d.correlationFunctions(n,mass2,ng,palst=palst,pblst=palst,iprt=0,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,palst,'cab0=',cab0[0]
      cab1 = pepo2cpeps(npepo,palst,palst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,palst,'cab1=',cab1[0]
      cab1 = pepo2cpeps(npepo2,palst,palst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,palst,'cab1=',cab1[0]

   print 'Case-4: right'
   for i in range(k):
      palst = [(m-i,m-i)]
      pblst = [(m+i,m+i+1)]
      # Comparison   
      cab0 = num2d.correlationFunctions(n,mass2,ng,palst=palst,pblst=pblst,iprt=0,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0[0]
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0]
      cab1 = pepo2cpeps(npepo2,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0]

   print 'Case-3: upper-right'
   for i in range(k):
      palst = [(m-i,m-i)]
      pblst = [(m+i+1,m+i+1)]
      # Comparison   
      cab0 = num2d.correlationFunctions(n,mass2,ng,palst=palst,pblst=pblst,iprt=0,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0[0]
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0]
      cab1 = pepo2cpeps(npepo2,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0]

   print 'Case-2: above'
   for i in range(k):
      palst = [(m-i,m-i)]
      pblst = [(m+i+1,m+i)]
      # Comparison   
      cab0 = num2d.correlationFunctions(n,mass2,ng,palst=palst,pblst=pblst,iprt=0,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0[0]
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0]
      cab1 = pepo2cpeps(npepo2,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0]

   print 'Case-1: upper-left'
   for i in range(k):
      palst = [(m-i,m-i)]
      pblst = [(m+i-1,m+i+1)]
      # Comparison   
      cab0 = num2d.correlationFunctions(n,mass2,ng,palst=palst,pblst=pblst,iprt=0,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab0=',cab0[0]
      cab1 = pepo2cpeps(npepo,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0]
      cab1 = pepo2cpeps(npepo2,palst,pblst,auxbond=abond)
      print 'i=',i,'pa,pb=',palst,pblst,'cab1=',cab1[0]
