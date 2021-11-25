#!/usr/bin/env python3

"""Multigrid solver for Poisson equation."""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from mpi4py import MPI
from multigrid_module import split, laplacian, smooth_sweep, smooth, coarse, interpolate_add_to


class Buffers:
    """Set the MPI buffers for the variable var."""

    def __init__(self, shape, var, width, overlap=0):
        """
        Width of the buffer.

        overlap : width of the overlap of the interior domains
        """
        self.comm = MPI.COMM_WORLD
        self.var = var
        self.rank = self.comm.Get_rank()
        self.send_buffers = {}
        self.rec_buffers = {}
        self.width = width
        self.neighbours = {}
        self.ny, self.nx = shape
        self.send_slices = {}
        self.rec_slices = {}

        self.send_slices["east"] = np.s_[:, -2 * width - overlap: -width - overlap]
        self.send_slices["west"] = np.s_[:, width + overlap: overlap + 2 * width]
        self.send_slices["north"] = np.s_[-2 * width - overlap: -width -overlap, :]
        self.send_slices["south"] = np.s_[width + overlap: overlap + 2 * width, :]

        self.rec_slices["east"] = np.s_[:, -width:]
        self.rec_slices["west"] = np.s_[:, :width]
        self.rec_slices["north"] = np.s_[-width:, :]
        self.rec_slices["south"] = np.s_[:width, :]

        if self.rank == 0:
            self.neighbours = {"east": 1, "north": 2}
        if self.rank == 1:
            self.neighbours = {"west": 0, "north": 3}
        if self.rank == 2:
            self.neighbours = {"east": 3, "south": 0}
        if self.rank == 3:
            self.neighbours = {"west": 2, "south": 1}

        for direction in self.neighbours.keys():
            if direction == "west" or direction == "east":
                self.send_buffers[direction] = np.zeros((self.ny, self.width))
                self.rec_buffers[direction] = np.zeros((self.ny, self.width))
            elif direction == "north" or direction == "south":
                self.send_buffers[direction] = np.zeros((self.width, self.nx))
                self.rec_buffers[direction] = np.zeros((self.width, self.nx))

        self.reqr = []
        self.reqs = []

        for direc, ngbr_rank in self.neighbours.items():
            sr = self.comm.Send_init(self.send_buffers[direc], ngbr_rank, tag=1)
            rr = self.comm.Recv_init(self.rec_buffers[direc], ngbr_rank, tag=1)
            self.reqr += [rr]
            self.reqs += [sr]

    def fill_buffers(self):
        """Fill the communication buffers."""
        send = self.send_buffers
        slices = self.send_slices
        MPI.Prequest.Startall(self.reqr)
        for direction in self.neighbours.keys():
            send[direction][:] = self.var[slices[direction]]
        MPI.Prequest.Startall(self.reqs)
        MPI.Prequest.Waitall(self.reqs)

    def fill_var(self):
        """Fill the communication buffers."""
        rec = self.rec_buffers
        slices = self.rec_slices
        MPI.Prequest.Waitall(self.reqr)
        for direction in self.neighbours.keys():
            self.var[slices[direction]] = rec[direction]


def smooth_parallel(b, x, h, lplc, r, shape, iterations, com):
    """loop."""
    ny, nx = shape

    com.fill_buffers()
    com.fill_var()
    for i in range(iterations):
        smooth_sweep(b, x, h)
        com.fill_buffers()
        com.fill_var()

    laplacian(x, lplc, h)
    # Computing the residual.
    r[:] = b - lplc



def gather_blocks(comm, M_block, M_full, rank, overlap=2):
    _, nx = M_block.shape
    nx2 = nx - 1
    _, nx0 = M_full.shape
    # assert nx2*2 == nx0 + 1

    M_blocks = []
    for i in range(4):
        if i == rank:
            M_blocks += [M_block]
        else:
            M_blocks += [np.zeros_like(M_block)]

    for i in range(4):
        comm.Bcast(M_blocks[i], i)

    M_full[:nx2, :nx2] = M_blocks[0][:-1, :-1]
    M_full[:nx2, nx2:] = M_blocks[1][:-1, overlap:]
    M_full[nx2:, :nx2] = M_blocks[2][overlap:, :-1]
    M_full[nx2:, nx2:] = M_blocks[3][overlap:, overlap:]

    # if rank == 0:
    #     plt.pcolormesh(M_full)
    #     plt.show()

class Multigrid:
    def __init__(self, b0, x0, r0, h0, epsilon, n):
        _, self.nx = x0.shape
        self.n = n
        self.x = []
        self.b = []
        self.lplc = []
        self.r = []
        self.h = []
        self.b0 = b0
        self.x0 = x0
        self.r0 = r0
        self.h0 = h0
        self.epsilon = epsilon
        for i in range(1, n+1):
            self.x += [np.zeros((2 ** i + 1, 2 ** i + 1))]
            self.b += [np.zeros((2 ** i + 1, 2 ** i + 1))]
            self.lplc += [np.zeros((2 ** i + 1, 2 ** i + 1))]
            self.r += [np.zeros((2 ** i + 1, 2 ** i + 1))]
            self.h += [h0 * 2 ** (n + 1 - i)]

        self.x += [x0]
        self.b += [b0]
        self.lplc += [np.zeros_like(b0)]
        self.r += [r0]
        self.h += [h0]

        self.temp = np.zeros((2**(n-1)+2, (2**(n-1)+2)))
        self.buff = Buffers(x0.shape, x0, 1, 3)

    def solve(self):
        """
        Solve the Poisson equation by reapeating v cycles.
        """
        it = 0
        status = 0
        err_old = 1000
        fail = False
        self.b0[0, :] = 0
        self.b0[-1, :] = 0
        self.b0[:, -1] = 0
        self.b0[:, 0] = 0
        x0 = self.x0
        n1 = 10
        n2 = 20
        x = self.x

        # Iterate until the residual is smaller than epsilon.
        smooth_parallel(self.b0, x0, self.h0, self.lplc[-1], self.r0, x0.shape, n1, self.buff)
        while status < 4 and it < 8:
            coarse(self.r0, self.temp)
            gather_blocks(self.buff.comm, self.temp, self.b[-2], self.buff.rank)
            for i in range(2, self.n+1):
                smooth(self.b[-i], x[-i], self.h[-i], self.lplc[-i], self.r[-i], n2)
                coarse(self.r[-i], self.b[-i - 1])

            smooth(self.b[0], x[0], self.h[0], self.lplc[0], self.r[0],  n2)
            for i in range(1, self.n):
                interpolate_add_to(x[i - 1], x[i])
                smooth(self.b[i], x[i], self.h[i], self.lplc[i], self.r[i], n2)

            split(self.buff.rank, x[-2], self.temp)
            interpolate_add_to(self.temp, x[-1])
            smooth_parallel(self.b0, x0, self.h0, self.lplc[-1], self.r0, x0.shape, n1, self.buff)
            self.buff.comm.barrier()
            err = np.amax(np.abs(self.r0))
            if self.buff.rank==0:
                if err > err_old:
                    print("error")

            if err > err_old:
                fail = True

            err_old = err
            if err > self.epsilon and not fail:
                status = self.buff.comm.allreduce(0)
            else:
                status = self.buff.comm.allreduce(1)
            it += 1
        if self.buff.rank == 0:
            print("v cycles :", it)
        self.buff.comm.barrier()


def test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n = 8
    epsilon = 0.001
    nx1 = 2 ** n + 1
    nx0 = 2 ** (n + 1) + 1
    nx1_sub = nx1 + 2
    b_max = 50
    print(f"nx = {nx1_sub}")
    x = np.linspace(-10, 10, nx0)
    y = np.linspace(-10, 10, nx0)
    h = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    xr = [ 4, -2,  6,  3,  3,  5, -5, -7]
    yr = [-2,  6,  6, -5,  5, -5, -2, -2]
    print(yr)
    r = (X-x) ** 2 + (Y-y) ** 2
    b0 = X*0
    sign = 1
    for x, y in zip(xr, yr):
        r = (X-x) ** 2 + (Y-y) ** 2
        b0 += sign*b_max * np.exp(-r * 7)
        sign *= -1

    b = np.zeros((nx1_sub, nx1_sub))
    split(rank, b0, b)

    a = np.zeros_like(b)
    R = np.zeros_like(a)

    poisson = Multigrid(b, a, R, h, epsilon, n)

    # t0 = perf_counter()
    poisson.solve()
    # t1 = perf_counter()
    # print(f"time multi 1    {t1-t0} s")

    t = 0
    for i in range(10):
        a[:] = 0
        t0 = perf_counter()
        poisson.solve()
        t1 = perf_counter()
        t += t1-t0
    print(f"time multi 1    {t/10} s")

    # a_full = np.zeros_like(b0)
    # R_full = np.zeros_like(b0)
    # gather_blocks(comm, a, a_full, rank, 4)
    # gather_blocks(comm, R, R_full, rank, 4)

    # if rank==0:
    #     fig, ax = plt.subplots(1, 2)
    #     ax0, ax1 = ax
    #     ax0.pcolormesh(a_full)
    #     r_max = np.amax(np.abs(R_full/b_max))
    #     cm = ax1.pcolormesh(R_full / b_max, cmap="bwr", vmin=-r_max, vmax=r_max)
    #     fig.suptitle(f"{nx0}x{nx0} grid points")
    #     ax0.set_title("Phi")
    #     ax1.set_title("Residual / max(B)")
    #     plt.colorbar(cm)
    #     plt.savefig("test.png")
    #     plt.show()

if __name__ == "__main__":
    test()
