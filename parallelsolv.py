#!/usr/bin/env python3
"""
j
"""
import numpy as np
from numba import jit
from time import perf_counter
from mpi4py import MPI


class Buffers:
    def __init__(self, shape, var, width):
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

        self.send_slices["east"] = np.s_[:, -2 * width : -width]
        self.send_slices["west"] = np.s_[:, width : 2 * width]
        self.send_slices["north"] = np.s_[-2 * width : -width, :]
        self.send_slices["south"] = np.s_[width : 2 * width, :]

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


@jit(nopython=True, cache=True)
def iterate(a, a_new, ny, nx):
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            a_new[j, i] = 0.25 * (
                a[j, i + 1] + a_new[j, i - 1] + a[j + 1, i] + a_new[j - 1, i] + b[j, i]
            )


def smooth_parallel(b, a, a_new, shape, epsilon, com):
    """loop."""
    ny, nx = shape
    err = epsilon + 1
    it = 0
    sum_err = epsilon * 4 + 1
    comm = MPI.COMM_WORLD

    while sum_err > epsilon * 4:
        com.fill_buffers()
        com.fill_var()
        iterate(a, a_new, ny, nx)
        err = np.amax(np.abs(a[1:-1, 1:-1] - a_new[1:-1, 1:-1]))
        sum_err = comm.allreduce(err)
        a[:] = a_new
        it += 1


def poisson_parallel_multigrid(b, a, shape, epsilon, n):
    """
    Solve the Poisson equation using Gauss-Seidel Method.
    TODO this is realy slow
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ny, nx = shape
    a_new = a.copy()
    a_crs = []
    a_crs_new = []
    bs = []
    coms = []
    for i in range(1, n):
        a_crs += [np.zeros((2 ** i + 1, 2 ** i + 1))]
        a_crs_new += [np.zeros((2 ** i + 1, 2 ** i + 1))]
        com = communications((2 ** i + 1, 2 ** i + 1), a_crs_new[i - 1], 1)
        coms += [com]
        bs += [np.zeros((2 ** i + 1, 2 ** i + 1))]

    com = communications(shape, a_new, 1)
    coms += [com]
    a_crs += [a]
    a_crs_new += [a_new]
    bs += [b]

    for i in range(n - 2, -1, -1):
        t0 = perf_counter()
        reduce(bs[1 + i], bs[i])
        t1 = perf_counter()
    print(f"time rduce {(t1-t0)} s", flush=True)

    for i in range(0, n - 1):
        t0b = perf_counter()
        gauss_seidel(bs[i], a_crs[i], a_crs_new[i], a_crs[i].shape, 0.001, coms[i])
        t0 = perf_counter()
        interpolate(a_crs[i], a_crs[i + 1])
        t1 = perf_counter()
    print(f"time gauss {(t0-t0b)} s", flush=True)
    print(f"time interp {(t1-t0)} s", flush=True)
    # if rank == 0:
    #     plt.pcolormesh(a_crs[i])
    #     plt.show()

    t0 = perf_counter()
    gauss_seidel(b, a, a_new, shape, epsilon, coms[n - 1])
    t1 = perf_counter()
    print(f"time final gauss {t1-t0} s", flush=True)


def poisson_parallel(b, a, shape, epsilon, n):
    """
    Solve the Poisson equation using Gauss-Seidel Method.
    """
    ny, nx = shape
    a_new = a.copy()

    com = communications(shape, a_new, 1)

    gauss_seidel(b, a, a_new, shape, epsilon, com)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n = 6
    nx = 2 ** n + 1
    ny = nx
    x = np.linspace(-10, 10, 2 * nx)
    y = np.linspace(-10, 10, 2 * ny)
    X, Y = np.meshgrid(x, y)
    r = X ** 2 + Y ** 2
    b0 = np.exp(-r / 8)
    b = np.zeros((ny + 1, nx + 1))
    sol = X ** 2
    if rank == 0:
        b[:] = b0[0 : ny + 1, 0 : nx + 1]
    elif rank == 1:
        b[:] = b0[0 : ny + 1, nx - 1 :]
    elif rank == 2:
        b[:] = b0[ny - 1 :, 0 : nx + 1]
    elif rank == 3:
        b[:] = b0[ny - 1 :, nx - 1 :]

    a = np.zeros_like(b)
    t0 = perf_counter()
    poisson_parallel(b, a, (ny + 1, nx + 1), 0.01, n)
    t1 = perf_counter()
    print(f"time parallel {t1-t0} s", flush=True)
    b = b * 1.1
    t0 = perf_counter()
    poisson_parallel(b, a, (ny + 1, nx + 1), 0.01, n)
    t1 = perf_counter()
    print(f"time parallel {t1-t0} s", flush=True)
    # if rank == 0:
    #     plt.pcolormesh(a)
    #     plt.show()
