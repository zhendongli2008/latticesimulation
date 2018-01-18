import num2d

mass2c = -1.730814685786978

def test():
   m = 50 #100
   n = 2*m+1
   # Exact
   # ss_tc = [0.63662]
   # Approximate
   ng = 2
   mass2 = mass2c #mass2c #6 #mass2c
   beta = 1.0/(4.0+mass2)
   vapp = []
   print '(m,n)=',(m,n),'mass2=',mass2,'beta=',beta,'T=',1.0/beta
   for j in range(1,2):
      pb = (m+j,m+j)
      cij = num2d.genVpeps(n,mass2=mass2,ng=ng,pa=(m,m),pb=pb)
      print 'j=',j,'pb=',pb,'cij=',cij,'cij/beta=',cij/beta
      vapp.append(cij/beta)
   print 'vapp=',vapp
   return 0

if __name__ == '__main__':
   test()
