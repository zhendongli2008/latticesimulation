import numpy
import scipy.linalg
import contraction2d
import matplotlib.pyplot as plt
import num2d

#WRONG:
#ss_tc = [0.63662, 0.54038, 0.489268, 0.455647, 0.431072, 0.411942, 0.396414, \
#		0.383427, 0.372321, 0.362655, 0.354126, 0.346513, 0.339653, 0.333422, \
#		0.327724, 0.322482, 0.317633, 0.313128, 0.308926, 0.304991]
#ss_th = [0.63662, 0.405504, 0.343791, 0.292135, 0.264268, 0.239481, 0.222839, \
#		0.207697, 0.196337, 0.185896, 0.177508, 0.169761, 0.163239, 0.157203, \
#		0.151942, 0.147065, 0.142698, 0.138661, 0.13497, 0.131553]
#ss_tl = [0.673553, 0.596592, 0.559762, 0.537758, 0.523068, 0.512582, 0.504751, \
#		0.498711, 0.493936, 0.490088, 0.486939, 0.484328, 0.482141, 0.480292, \
#		0.478717, 0.477365, 0.476198, 0.475186, 0.474303, 0.47353]

def test():
   m = 100
   n = 2*m+1
   # Exact
   #>plt.plot(ss_tc,'ro-',label='T=Tc')
   #>plt.plot(ss_th,'go-',label='T=10')
   #>plt.plot(ss_tl,'bo-',label='T=1/0.445')
   # Approximate
   ng = 2
   mass2 = -1.9 #1.0 #-1.730814685786978
   beta = 1.0/(4.0+mass2)
   vapp = []
   print '(m,n)=',(m,n),'mass2=',mass2,'beta=',beta
   for j in range(1,5):
      pb = (m+j,m+j)
      cij = num2d.genVpeps(n,mass2=mass2,ng=ng,pa=(m,m),pb=pb)
      print 'j=',j,'pb=',pb,'cij=',cij,'cij/beta=',cij/beta
      vapp.append(cij/beta)
   plt.plot(vapp,'b--',label='mass2='+str(mass2))
   # Comparison
   plt.legend()
   plt.show()
   return 0

if __name__ == '__main__':
   test()
