#!/usr/bin/env python3

"""Multigrid solver for Poisson equation."""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from mpi4py import MPI
from multigrid_module import (laplacian, smooth_sweep, smooth_sweep_back,
                              split, smooth, coarse, interpolate_add_to)


class Buffers:
    """Set the MPI buffers for the variable var."""

    def __init__(self, shape, var, width, overlap=1):
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


    for i in range(iterations//2):
        com.fill_buffers()
        com.fill_var()
        smooth_sweep(b, x, h)
        com.fill_buffers()
        com.fill_var()
        smooth_sweep_back(b, x, h)

    laplacian(x, lplc, h)
    # Computing the residual.
    r[1:-1, 1:-1] = b[1:-1, 1:-1] - lplc[1:-1, 1:-1]



def gather_blocks(comm, M_block, M_full):
    _, nx = M_block.shape
    nx2 = nx - 1
    _, nx0 = M_full.shape
    assert nx2*2 == nx0 + 1
    rank = comm.Get_rank()

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


def offsets(rank, width=1):
    ofs = [{"j": 0, "i": 0}, {"j": 0, "i": -width},
           {"j": -width, "i": 0}, {"j": -width, "i": -width}]
    return ofs[rank]


class Multigrid:
    def __init__(self, b0, x0, r0, h0, epsilon, n, n_para=1):
        # IDEA computing R directly whitout storing lplc
        _, self.nx = x0.shape
        self.n = n
        assert self.nx == 2**n+2
        self.n_para = n_para
        assert n_para < n

        # Whole domain
        self.x_wl = []
        self.b_wl = []
        self.h_wl = []
        self.r_wl = []
        self.lplc_wl = []

        # Subdomain
        self.b_sb = []
        self.x_sb = []
        self.h_sb = []
        self.r_sb = []
        self.lplc_sb = []

        self.b0 = b0
        self.x0 = x0
        self.r0 = r0
        self.h0 = h0
        self.epsilon = epsilon
        self.bufs = []
        self.width_halo = 1

        for i in range(n_para-1, n):
            self.lplc_wl += [np.zeros((2**(n-i)+1, 2**(n-i)+1))]
            self.x_wl += [np.zeros((2**(n-i)+1, 2**(n-i)+1))]
            self.b_wl += [np.zeros((2**(n-i)+1, 2**(n-i)+1))]
            self.r_wl += [np.zeros((2**(n-i)+1, 2**(n-i)+1))]
            self.h_wl += [h0 * 2 ** (i)]

        for i in range(n_para):
            if i == 0:
                self.x_sb += [x0]
                self.b_sb += [b0]
                self.r_sb += [r0]
                self.lplc_sb += [np.zeros((2**(n-i)+2, 2**(n-i)+2))]
            else:
                self.x_sb += [np.zeros((2**(n-i)+2, 2**(n-i)+2))]
                self.b_sb += [np.zeros((2**(n-i)+2, 2**(n-i)+2))]
                self.r_sb += [np.zeros((2**(n-i)+2, 2**(n-i)+2))]
                self.lplc_sb += [np.zeros((2**(n-i)+2, 2**(n-i)+2))]
            self.h_sb += [h0 * 2 ** i]
            self.bufs += [Buffers(self.x_sb[i].shape, self.x_sb[i],
                                  self.width_halo, 1)]

        self.b_sb += [np.zeros((2**(n-n_para)+2, 2**(n-n_para)+2))]
        # offset due to the hallos
        self.ofst = offsets(self.bufs[0].rank, self.width_halo)


    def solve(self):
        """
        Solve the Poisson equation by reapeating v cycles.
        """
        it = 0
        status = 0
        err_old = 100
        fail = False
        # self.b0[0, :] = 0
        # self.b0[-1, :] = 0
        # self.b0[:, -1] = 0
        # self.b0[:, 0] = 0
        x_sb = self.x_sb
        x_wl = self.x_wl
        b_sb = self.b_sb
        b_wl = self.b_wl
        lplc_sb = self.lplc_sb
        lplc_wl = self.lplc_wl
        r_sb = self.r_sb
        r_wl = self.r_wl
        h_wl = self.h_wl
        h_sb = self.h_sb
        n1 = 200
        n2 = 200
        ofst = self.ofst
        n_para = self.n_para
        n = self.n
        rank = self.bufs[0].rank
        comm = self.bufs[0].comm

        # Iterate until the residual is smaller than epsilon.
        while status < 4 and it < 200:
            # _________DESCENT_________

            # _SUBDOMAIN
            for i in range(n_para):
                smooth_parallel(b_sb[i], x_sb[i], h_sb[i], lplc_sb[i],
                                r_sb[i], x_sb[i].shape, n1, self.bufs[i])
                coarse(r_sb[i], b_sb[i+1], ofst["i"], ofst["j"])
                # plt.pcolormesh(b_sb[1+i])
                # plt.title(f"{rank}")
                # plt.colorbar()
                # plt.show()


            # _WHOLE DOMAIN
            gather_blocks(comm, b_sb[-1], b_wl[0])
            smooth(b_wl[0], x_wl[0], h_wl[0], lplc_wl[0], r_wl[0], n2)
            for i in range(1, n-n_para+1):
                coarse(r_wl[i-1], b_wl[i], 0, 0)
                smooth(b_wl[i], x_wl[i], h_wl[i], lplc_wl[i], r_wl[i], n2)
                # plt.pcolormesh(b_wl[i])
                # plt.title(f"{rank}")
                # plt.colorbar()
                # plt.show()

            # # # _________ASCENT_________

            # # _WHOLE DOMAIN
            for i in range(2, n-n_para+1):
                interpolate_add_to(x_wl[-i+1], x_wl[-i], 0, 0)
                smooth(b_wl[-i], x_wl[-i], h_wl[-i], lplc_wl[-i], r_wl[-i], n2)
                # plt.pcolormesh(x_wl[-i])
                # plt.title(f"ascent wl {rank}")
                # plt.colorbar()
                # plt.show()

            # _SUBDOMAIN
            split(x_wl[0], b_sb[-1], rank)
            # !!! b_sb has one more element than x_sb[]
            interpolate_add_to(b_sb[-1], x_sb[-1], ofst["i"], ofst["j"])
            smooth_parallel(b_sb[-2], x_sb[-1], h_sb[-1], lplc_sb[-1],
                            r_sb[-1], x_sb[-1].shape, n2, self.bufs[-1])
            # plt.pcolormesh(r_sb[-1])
            # plt.title(f"ascent wl {rank}")
            # plt.colorbar()
            # plt.show()

            for i in range(2, n_para+1):
                interpolate_add_to(x_sb[-i+1], x_sb[-i],
                                   ofst["i"], ofst["j"])

                # !!! b_sb has one more element than the other _sb[]
                smooth_parallel(b_sb[-i-1], x_sb[-i], h_sb[-i], lplc_sb[-i],
                                r_sb[-i], x_sb[-i].shape, n1, self.bufs[-i])
                # plt.pcolormesh(x_sb[-i])
                # plt.title(f"ascent wl {rank}")
                # plt.colorbar()
                # plt.show()

            self.bufs[0].comm.barrier()
            err = np.amax(np.abs(self.r_sb[0]))
            if rank == 0:
                if err > err_old*1.1:
                    print("error")

            if err > err_old*1.1:
                fail = True

            err_old = err
            if err > self.epsilon and not fail:
                status = comm.allreduce(0)
            else:
                status = comm.allreduce(1)
            it += 1
        if rank == 0:
            print("v cycles :", it)
        self.bufs[0].comm.barrier()


def test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n = 7
    epsilon = 0.001
    nx1 = 2 ** n + 1
    nx0 = 2 ** (n + 1) + 1
    b_max = 50
    print(f"nx = {nx1}")
    x = np.linspace(-10, 10, nx0)
    y = np.linspace(-10, 10, nx0)
    h = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    r = (X) ** 2 + (Y) ** 2
    # b0 = b_max*np.exp(-r/4)
    b0 = np.zeros_like(X)
    sign = 1
    xr = [ 4, -2,  6,  3,  3,  5, -5, -7]
    yr = [-2,  6,  6, -5,  5, -5, -2, -2]
    for x, y in zip(xr, yr):
        r = (X-x) ** 2 + (Y-y) ** 2
        b0 += sign*b_max * np.exp(-r * 7)
        sign *= -1

    b = np.zeros((nx1+1, nx1+1))
    split(b0, b, rank)

    a = np.zeros_like(b)
    R = np.zeros_like(a)

    poisson = Multigrid(b, a, R, h, epsilon, n, 2)
    poisson.solve()

    # lplc = np.zeros_like(a)
    # buf = Buffers(a.shape, a, 1, 1)
    # smooth_parallel(b, a, h, lplc, R, a.shape, 500, buf)

    # t = 0
    # for i in range(10):
    #     a[:] = 0
    #     t0 = perf_counter()
    #     poisson.solve()
    #     t1 = perf_counter()
    #     t += t1-t0
    # print(f"time multi 1    {t/10} s")
    plt.pcolormesh(R/b_max)
    plt.colorbar()
    plt.show()

    # a_full = np.zeros_like(b0)
    # R_full = np.zeros_like(b0)
    # gather_blocks(comm, a, a_full)
    # gather_blocks(comm, R, R_full)

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
