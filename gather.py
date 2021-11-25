#!/usr/bin/env python3
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpi4py import MPI
from parallel_multigrid import Buffers

def interpolate_add_to(a, a_new):
    """Interpolate."""
    for j in range(1, a.shape[0] - 1):
        for i in range(1, a.shape[1] - 1):
            a_new[2 * j, 2 * i] += a[j, i]
    for j in range(0, a.shape[0] - 1):
        for i in range(0, a.shape[1] - 1):
            a_new[2 * j + 1, 2 * i + 1] += (
                a[j + 1, i + 1] + a[j + 1, i] + a[j, i + 1] + a[j, i]
            ) / 4
    for j in range(1, a.shape[0] - 1):
        for i in range(0, a.shape[1] - 1):
            a_new[2 * j, 2 * i + 1] += (a[j, i] + a[j, i + 1]) / 2

    for j in range(0, a.shape[0] - 1):
        for i in range(1, a.shape[1] - 1):
            a_new[2 * j + 1, 2 * i] += (a[j, i] + a[j + 1, i]) / 2


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

    depth_para = 3
    n = 7
    nx0 = 2**(1+n) + 1

    # nx1_sub = 2**n + 2
    # nxi = nx1_sub-2

    x = np.linspace(-5, 5, nx0)# +6-2*rank
    X, Y = np.meshgrid(x, x)
    R = X**2+Y**2
    A = np.exp(-R/4)
    A[0, :] = 0
    A[-1, :] = 0
    A[:, -1] = 0
    A[:, 0] = 0

    whole = [np.zeros((2**(n-i)+1, 2**(n-i)+1)) for i in range(depth_para-1, n)]
    sub = [np.zeros((2**(n-i)+2, 2**(n-i)+2)) for i in range(depth_para+1)]
    buf = [Buffers(sub[i].shape, sub[i], 1, 1) for i in range(depth_para)]
    # sub_temp = np.zeros((2**(n-depth_para)+2, 2**(n-depth_para)+2))

    split(rank, A, sub[0])

    for i in range(depth_para):
        # smooth ...
        buf[i].fill_buffers()
        buf[i].fill_var()
        buf[i].fill_buffers()
        buf[i].fill_var()
        coarse(sub[i], sub[i+1])
        if rank == 3:
            plt.pcolormesh(sub[i+1])
            plt.title(f"sub {i+1}")
            plt.colorbar()
            plt.show()

    gather_blocks(comm, sub[-1], whole[0])
    if rank == 3:
        plt.pcolormesh(whole[0])
        plt.title("whole 0")
        plt.colorbar()
        plt.show()

    for i in range(n-depth_para):
        coarse(whole[i], whole[i+1])
        if rank == 3:
            plt.pcolormesh(whole[i+1])
            plt.title(f"whole {i+1}")
            plt.colorbar()
            plt.show()

    for i in range(1, n-depth_para+1):
        whole[-i-1][:] = 0
        interpolate_add_to(whole[-i], whole[-i-1])
        if rank == 3:
            plt.pcolormesh(whole[-i-1])
            plt.title(f"whole {len(whole)-i-1}")
            plt.colorbar()
            plt.show()


    # sub3_temp = np.zeros((nx3+1, nx3+1))
    # split(rank, sub2_whole, sub3_temp)
    # sub2_old = sub2.copy()
    # sub2[:] = 0
    # interpolate_add_to(sub3_temp, sub2)
