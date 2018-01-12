import numpy
import scipy.linalg
import matplotlib.pyplot as plt

a = 0.01
kappa = 100.0
ka = kappa*a
mass = ka**2

n = 1000 # For small kappa, large n is required 20000
tij = numpy.diag([2.0+mass]*n)
for i in range(n-1):
   tij[i,i+1] = tij[i+1,i] = -1.0
kbd = scipy.linalg.inv(tij)[0,0]
print 'kbd(num)=',kbd
kbd = 1.0+0.5*ka**2-0.5*ka*numpy.sqrt(4+ka**2)
print 'kbd(ana)=',kbd

n = 1000
bc = 'OBC'

tij = numpy.diag([2.0+mass]*n)
for i in range(n-1):
   tij[i,i+1] = tij[i+1,i] = -1.0

tij[0,0] += -kbd
tij[n-1,n-1] += -kbd 
print numpy.diag(tij)

plt.matshow(tij)
plt.show()

e,v=scipy.linalg.eigh(tij)
print 'e=',e
tinv = 2.0*kappa*a*scipy.linalg.inv(tij)
print numpy.diag(tinv)
plt.matshow(tinv)
plt.show()

xa = numpy.arange(0,n)*a

style = '-'
plt.plot(xa,tinv[0]    ,'r'+style)
plt.plot(xa,tinv[1]    ,'k'+style)
plt.plot(xa,tinv[9]    ,'r'+style)
plt.plot(xa,tinv[n/4]  ,'g'+style)
plt.plot(xa,tinv[3*n/4],'g'+style)
plt.plot(xa,tinv[n/2]  ,'b'+style)
plt.plot(xa,tinv[n-1]  ,'b'+style)
plt.ylim([0,1.1])
plt.show()
exit()

# 2D
iden = numpy.identity(n)
mat = numpy.einsum('ij,ab->iajb',tij,iden)\
    + numpy.einsum('ij,ab->iajb',iden,tij)
mat = mat.reshape((n*n,n*n))
plt.matshow(mat)
plt.show()
matinv = scipy.linalg.inv(mat)
plt.matshow(matinv)
plt.show()

plt.plot(matinv[0,:n],'ro-')
plt.plot(matinv[0,n:2*n],'ko-')
plt.plot(matinv[0,[i*n for i in range(n)]],'ko-')
plt.plot(matinv[1,:n],'ko-')
plt.plot(matinv[9,:n],'ro-')
plt.plot(matinv[n/4,:n],'go-')
plt.plot(matinv[3*n/4,:n],'go-')
plt.plot(matinv[n/2,:n],'bo-')
plt.plot(matinv[n-1,:n],'bo-')
plt.show()

# 3D
mat = numpy.einsum('ij,ab,xy->iaxjby',tij,iden,iden) \
    + numpy.einsum('ij,ab,xy->iaxjby',iden,tij,iden) \
    + numpy.einsum('ij,ab,xy->iaxjby',iden,iden,tij)
mat = mat.reshape((n*n*n,n*n*n))
plt.matshow(mat)
plt.show()
matinv = scipy.linalg.inv(mat)
plt.matshow(matinv)
plt.show()

plt.plot(matinv[0,:n],'ro-')
plt.plot(matinv[0,n:2*n],'ko-')
plt.plot(matinv[0,[i*n for i in range(n)]],'ko-')
plt.plot(matinv[1,:n],'ko-')
plt.plot(matinv[9,:n],'ro-')
plt.plot(matinv[n/4,:n],'go-')
plt.plot(matinv[3*n/4,:n],'go-')
plt.plot(matinv[n/2,:n],'bo-')
plt.plot(matinv[n-1,:n],'bo-')
plt.show()
