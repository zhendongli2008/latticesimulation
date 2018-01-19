import numpy
from isingMapping import mass2c
import matplotlib.pyplot as plt
import genFit

ng = 2
n = 13
center = (n/2,n/2)
mass2lst = genFit.genMass2lst(mass2c,50,20)

info = ['tmp',ng,n,center,mass2lst]
genFit.checkData(info)
exit()
clst = genFit.fitCoulomb(info)
