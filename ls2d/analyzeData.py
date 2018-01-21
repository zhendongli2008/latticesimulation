import numpy
from isingMapping import mass2c
import matplotlib.pyplot as plt
import genFit

dirname = 'tmp2'
ng = 2
n = 101
center = (n/2,n/2)
mass2lst = genFit.genMass2lst(mass2c,50,28)

info = [dirname,ng,n,center,mass2lst]
genFit.checkData(info,iop=0,thresh=20.0)
genFit.fitCoulomb(info,k=10,nselect=15,ifplot=True,\
		  skiplst=[])
