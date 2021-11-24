#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

nx = 9

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
A = np.ones((nx+1,nx+1))*rank+1
A[-1, -1] = 0
B = np.zeros((nx+1)**2*4)
comm.Allgather(A, B)

B2 = np.empty((2*nx, 2*nx))
B2[:nx, :nx] = B[0:nx**2].reshape(nx+1, nx)[:-1,:-1]
B2[:nx, nx:] = B[nx**2:nx**2*2].reshape(nx, nx)[:-1, 1:]
B2[nx:, :nx] = B[2*nx**2: 3*nx**2].reshape(nx, nx)[1:, :-1]
B2[nx:, nx:] = B[3*nx**2: 4*nx**2].reshape(nx, nx)[1:, 1:]

if rank == 0:
    plt.pcolormesh(B2)
    plt.show()
