#!/usr/bin/env python3
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpi4py import MPI

@jit(nopython=True, cache=True)
def coarse(a, a_crs):
    """Reduction on coarser grid."""
    for j in range(1, a_crs.shape[0] - 1):
        a_left = a[2 * j, 1] / 8 + (a[2 * j + 1, 1] + a[2 * j - 1, 1]) / 16
        for i in range(1, a_crs.shape[1] - 1):
            a_right = (
                a[2 * j, 2 * i + 1] / 8
                + (a[2 * j + 1, 2 * i + 1] + a[2 * j - 1, 2 * i + 1]) / 16
            )
            a_crs[j, i] = (
                a[2 * j, 2 * i] / 4
                + (a[2 * j + 1, 2 * i] + a[2 * j - 1, 2 * i]) * 1 / 8
            )
            a_crs[j, i] += a_right + a_left
            a_left = a_right


@jit(nopython=True, cache=True)
def split(rank, A_in, A_out):
    nx_out, _ = A_out.shape
    if rank == 0:
        A_out[:] = A[0: nx_out, 0: nx_out]
    elif rank == 1:
        A_out[:] = A[0: nx_out, -nx_out:]
    elif rank == 2:
        A_out[:] = A[-nx_out:, 0: nx_out]
    elif rank == 3:
        A_out[:] = A[-nx_out:, -nx_out:]



def gather_blocks(comm, M_block, M_full):
    _, nx = M_block.shape
    nx2 = nx - 1
    _, nx0 = M_full.shape
    assert nx2*2 == nx0 + 1

    M_blocks = []
    for i in range(4):
        if i == rank:
            M_blocks += [M_block]
        else:
            M_blocks += [np.zeros_like(M_block)]

    for i in range(4):
        comm.Ibcast(M_blocks[i], i)
    comm.barrier()

    M_full[:nx2, :nx2] = M_blocks[0][:-1, :-1]
    M_full[:nx2, nx2:] = M_blocks[1][:-1, 2:]
    M_full[nx2:, :nx2] = M_blocks[2][2:, :-1]
    M_full[nx2:, nx2:] = M_blocks[3][2:, 2:]


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n = 3
    nx0 = 2**(1+n) + 1
    nx1 = 2**n + 1
    nx1_sub = 2**n + 3
    nx2 = 2**(n-1) + 1
    nxi = nx1_sub-2

    x = np.arange(nx0)
    X, Y = np.meshgrid(x, x)
    A = X+Y
    A_sub = np.ones((nx1_sub, nx1_sub))
    split(rank, A, A_sub)

    # M_block += rank*4
    A_2 = np.zeros((nx2+1, nx2+1))
    coarse(A_sub, A_2)
    M_block = A_2+rank*2
    M_full = np.zeros((nx1, nx1))
    gather_blocks(comm, M_block, M_full)

    if rank == 0:
        plt.pcolormesh(M_full)
        plt.show()
