#!/usr/bin/env python3

"""Draft for parallel multigrid solver."""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpi4py import MPI
from parallel_multigrid import Buffers
from multigrid_module import interpolate_add_to, coarse, split


def gather_blocks(comm, M_block, M_full):
    _, nx = M_block.shape
    nx2 = nx - 1
    _, nx0 = M_full.shape
    assert nx2*2 == nx0 + 1
    rank = comm.Get_rank()
    print(type(rank))

    M_blocks = []
    for i in range(4):
        if i == rank:
            M_blocks += [M_block]
        else:
            M_blocks += [np.zeros_like(M_block)]

    for i in range(4):
        comm.Bcast(M_blocks[i], i)

    M_full[:nx2, :nx2] = M_blocks[0][:-1, :-1]
    M_full[:nx2, nx2:] = M_blocks[1][:-1, 2:]
    M_full[nx2:, :nx2] = M_blocks[2][2:, :-1]
    M_full[nx2:, nx2:] = M_blocks[3][2:, 2:]

def offsets(rank):
    ofs = [{"j": 0, "i": 0}, {"j": 0, "i": -1}, {"j": -1, "i": 0}, {"j": -1, "i": -1}]
    return ofs[rank]

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n_para = 3
    n = 4
    nx0 = 2**(1+n) + 1

    # nx1_sub = 2**n + 2
    # nxi = nx1_sub-2

    x = np.arange(nx0)
    X, Y = np.meshgrid(x, x)
    A = np.zeros_like(X, dtype=np.double)
    A[:] = X+Y
    A[0, :] = 0
    A[-1, :] = 0
    A[:, -1] = 0
    A[:, 0] = 0

    ofst = offsets(rank)

    whole = [np.zeros((2**(n-i)+1, 2**(n-i)+1)) for i in range(n_para-1, n)]
    sub = [np.zeros((2**(n-i)+2, 2**(n-i)+2)) for i in range(n_para+1)]
    buf = [Buffers(sub[i].shape, sub[i], 1, 1) for i in range(n_para+1)]
    # sub[-1] is a temporary buffer to store data before gathering or spliting

    split(A, sub[0], rank)
    buf[0].fill_buffers()
    buf[0].fill_var()
    buf[0].fill_buffers()
    buf[0].fill_var()

    ###############################
    #           Descent           #
    ###############################

    # _________Sub domain
    for i in range(n_para):
        buf[i].fill_buffers()
        buf[i].fill_var()
        buf[i].fill_buffers()
        buf[i].fill_var()
        coarse(sub[i], sub[i+1], ofst["i"], ofst["j"])
        # No fill because gather
        # smooth ...

    # _________Whole domain
    #
    gather_blocks(comm, sub[-1], whole[0])
    if rank == 3:
        plt.pcolormesh(whole[0])
        plt.title(f"sub {len(sub)-i-1}")
        plt.colorbar()
        plt.show()
    for i in range(n-n_para):
        coarse(whole[i], whole[i+1], 0, 0)

    ###############################
    #           Ascent            #
    ###############################
    for i in range(1, n-n_para+1):
        whole[-i-1][:] = 0
        interpolate_add_to(whole[-i], whole[-i-1], 0, 0)

    # _________Sub domain
    split(whole[0], sub[-1], rank)
    for i in range(1, n_para):
        interpolate_add_to(sub[-i], sub[-i-1], ofst["i"], ofst["j"])
        # if rank == 3:
        #     plt.pcolormesh(sub[-i-1])
        #     plt.title(f"sub {len(sub)-i-1}")
        #     plt.colorbar()
        #     plt.show()

    A[:] = 0
    gather_blocks(comm, sub[0], A)
    # if rank == 3:
    #     plt.pcolormesh(A)
    #     plt.title(f"whole")
    #     plt.colorbar()
    #     plt.show()
