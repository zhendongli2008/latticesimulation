import numpy
from mpi4py import MPI
from isingMapping import mass2c
import genFit

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

ng = 2
n = 11
center = (5,5)
mass2lst = 1.0*numpy.arange(0,50,2)
mass2rank = [mass2lst[i] for i in range(len(mass2lst)) if i%size == rank]

# mpirun -n 4 python genData.py
for mass2 in mass2rank: 
   info = ['tmp',ng,n,center,mass2]
   genFit.genData(info)
