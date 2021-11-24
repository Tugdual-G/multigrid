#!/usr/bin/env python3

"""Multigrid solver for Poisson equation."""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from time import perf_counter
from mpi4py import MPI


class Buffers:
    """Set the MPI buffers for the variable var."""

    def __init__(self, shape, var, width):
        """Width of the buffer."""
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
def laplacian(a, lplc, h=1):
    """Compute the laplacian of a."""
    ny, nx = a.shape
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            lplc[j, i] = (
                a[j, i - 1] + a[j, i + 1] + a[j - 1, i] +
                a[j + 1, i] - 4 * a[j, i]
            ) / (h ** 2)


@jit(nopython=True, cache=True)
def smooth(b, a, h, lplc, r, shape):
    """Gauss-Seidel method for multigrid."""
    ny, nx = shape

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            a[j, i] = 0.25 * (
                a[j, i + 1]
                + a[j, i - 1]
                + a[j + 1, i]
                + a[j - 1, i]
                - h ** 2 * b[j, i]
            )
    # Go back in the opposite direction
    for j in range(ny - 2, 0, -1):
        for i in range(nx - 2, 0, -1):
            a[j, i] = 0.25 * (
                a[j, i + 1]
                + a[j, i - 1]
                + a[j + 1, i]
                + a[j - 1, i]
                - h ** 2 * b[j, i]
            )

    laplacian(a, lplc, h)
    # Computing the residual.
    r[:] = b - lplc


def smooth_parallel(b, a, h, lplc, r, shape, iterations, com):
    """loop."""
    ny, nx = shape

    for i in range(iterations):
        com.fill_buffers()
        com.fill_var()
        smooth(b, a, h, lplc, r, shape)

    com.fill_buffers()
    com.fill_var()

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


def gather_blocks(comm, M_block, M_full):

    ny, nx = M_block.shape
    ny0, nx0 = M_full.shape
    assert ny*2 == ny0
    assert nx*2 == ny0

    rank = comm.Get_rank()
    B = np.empty(nx*nx*4)
    comm.Allgather(M_block, B)

    M_full = np.empty((2*nx, 2*nx))
    M_full[:nx, :nx] = B[0:nx**2].reshape(nx, nx)
    M_full[:nx, nx:] = B[nx**2:nx**2*2].reshape(nx, nx)
    M_full[nx:, :nx] = B[2*nx**2: 3*nx**2].reshape(nx, nx)
    M_full[nx:, nx:] = B[3*nx**2: 4*nx**2].reshape(nx, nx)

    if rank == 0:
        plt.pcolormesh(M_full)
        plt.show()


def poisson_multigrid(b0, x0, r0, h0, epsilon, n):
    """
    Solve the Poisson equation by reapeating v cycles.
    """
    # on 4 core processors:
    # Smooth in parallel (h)
    # Reduce on coarse gird in parallel
    # Gather residual to all
    # End of parallel communications
    # continue v cycle on the global residual
    # go up in v cycle
    # reach level (h2)
    # smooth h2
    # split data
    # Interpolate to process subdomain in h
    # start parallel communications again
    # Smooth in parallel

    ny, nx = x0.shape
    x = []
    b = []
    lplc = []
    r = []
    h = []
    for i in range(1, n+1):
        x += [np.zeros((2 ** i + 1, 2 ** i + 1))]
        b += [np.zeros((2 ** i + 1, 2 ** i + 1))]
        lplc += [np.zeros((2 ** i + 1, 2 ** i + 1))]
        r += [np.zeros((2 ** i + 1, 2 ** i + 1))]
        h += [h0 * 2 ** (n - i)]

    x += [x0]
    b += [b0]
    lplc += [b0.copy()]
    r += [r0]
    h += [h0]
    temp_coarse = np.zeros_like(b[-3])

    buff = Buffers(x0.shape, x0, 1)

    err = epsilon + 1
    it = 0
    while err > epsilon:
        smooth_parallel(b0, x0, h, lplc[-1], r0, x0.shape, 3, buff)
        coarse(r[-1], temp_coarse)
        # Reduce to b[2]
        gather_blocks(buff.comm, temp_coarse, b[-2])
        # Iterate until the residual is smaller than epsilon.
        for i in range(2, n+1):
            smooth(b[-i], x[-i], h[-i], lplc[-i], r[-i], x[-i].shape, 2)
            coarse(r[-i], b[-i - 1])

        smooth(b[0], x[0], h[0], lplc[0], r[0], x[0].shape, 3)
        for i in range(1, n+1):
            interpolate_add_to(x[i - 1], x[i])
            smooth(b[i], x[i], h[i], lplc[i], r[i], x[i].shape, 3)

        err = np.amax(np.abs(r[-1][1:-1, 1:-1]))
        it += 1
    print("v cycles :", it)


def test():
    n = 6
    epsilon = 0.001
    nx = 2 ** n + 1
    print(f"nx = {nx}")
    ny = nx
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    h = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    r = X ** 2 + Y ** 2
    b = 50 * np.exp(-r * 4)
    r = (4 + X) ** 2 + Y ** 2
    b -= 50 * np.exp(-r * 4)
    r = (5 + X) ** 2 + (5 + Y) ** 2
    b += 50 * np.exp(-r * 4)
    r = (X - 5) ** 2 + (Y - 2) ** 2
    b -= 50 * np.exp(-r * 4)
    a = np.zeros_like(b)
    b0 = b.copy()
    b = b0 * 1.01

    R = np.zeros_like(a)
    t0 = perf_counter()
    poisson_multigrid(b0, a, R, h, epsilon, n)
    t1 = perf_counter()
    print(f"time multi 1    {t1-t0} s", flush=True)
    t0 = perf_counter()
    poisson_multigrid(b, a, R, h, epsilon, n)
    t1 = perf_counter()
    print(f"time multi 2    {t1-t0} s", flush=True)
    fig, ax = plt.subplots(1, 2)
    ax0, ax1 = ax
    ax0.pcolormesh(a)
    a_max = np.amax(np.abs(a))
    cm = ax1.pcolormesh(R / a_max, cmap="bwr")

    fig.suptitle("1024x1024 grid points")
    ax0.set_title("Phi")
    ax1.set_title("Residual / max(Phi)")
    plt.colorbar(cm)
    plt.show()

    a[:] = 0
    t0 = perf_counter()
    poisson(b0, a, R, h, epsilon)
    t1 = perf_counter()
    print(f"time poisson 1  {t1-t0} s", flush=True)
    t0 = perf_counter()
    poisson(b, a, R, h, epsilon)
    t1 = perf_counter()
    print(f"time poisson 2  {t1-t0} s", flush=True)
    plt.pcolormesh(a)
    plt.colorbar()
    plt.show()


def test_interpol():
    from matplotlib.image import imread

    n = 7
    nx = 2 ** n + 1
    img = imread("image.jpg")[::-1, ::-1, 1]
    A = img[20 : 20 + nx, 80 : 80 + nx]
    Ac1 = np.zeros((2 ** (n - 1) + 1, 2 ** (n - 1) + 1))
    Ac2 = np.zeros((2 ** (n - 2) + 1, 2 ** (n - 2) + 1))
    Ai = np.zeros_like(A)
    print(A.shape)
    coarse(A, Ac1)
    coarse(Ac1, Ac2)
    Ac1 *= 0
    interpolate_into(Ac2, Ac1)
    interpolate_into(Ac1, Ai)
    fig, ax = plt.subplots(1, 3)
    ax0, ax1, ax2 = ax
    ax0.pcolormesh(A)
    ax1.pcolormesh(Ac2)
    ax2.pcolormesh(Ai)
    plt.show()

test()

if __name__ == "__main__":

    test()
