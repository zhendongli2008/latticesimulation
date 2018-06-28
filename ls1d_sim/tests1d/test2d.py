import numpy
import scipy.linalg
import matplotlib.pyplot as plt

mass = 0.001

n = 40
bc = 'OBC'

tij = numpy.diag([2.0+mass]*n)
for i in range(n-1):
   tij[i,i+1] = tij[i+1,i] = -1.0
plt.matshow(tij)
plt.show()

tinv = scipy.linalg.inv(tij)
plt.matshow(tinv)
plt.show()

style = 'o-'
plt.plot(tinv[0]    ,'r'+style)
plt.plot(tinv[1]    ,'k'+style)
plt.plot(tinv[9]    ,'r'+style)
plt.plot(tinv[n/4]  ,'g'+style)
plt.plot(tinv[3*n/4],'g'+style)
plt.plot(tinv[n/2]  ,'b'+style)
plt.plot(tinv[n-1]  ,'b'+style)
plt.show()

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

plt.plot(matinv[n*n/2,n*n/2:n*n/2+n],'ro-')
#plt.plot(matinv[0,:n],'ro-')
#plt.plot(matinv[0,n:2*n],'ko-')
#plt.plot(matinv[n/2,[i*n for i in range(n)]],'ko-')
#plt.plot(matinv[1,:n],'ko-')
#plt.plot(matinv[9,:n],'ro-')
#plt.plot(matinv[n/4,:n],'go-')
#plt.plot(matinv[3*n/4,:n],'go-')
#plt.plot(matinv[n/2,:n],'bo-')
#plt.plot(matinv[n-1,:n],'bo-')
plt.show()
