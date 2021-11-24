#!/usr/bin/env python3

"""Multigrid solver for Poisson equation."""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from time import perf_counter
from mpi4py import MPI


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

@jit(nopython=True, cache=True)
def split(rank, A_in, A_out):
    nx_out, _ = A_out.shape
    if rank == 0:
        A_out[:] = A_in[0: nx_out, 0: nx_out]
    elif rank == 1:
        A_out[:] = A_in[0: nx_out, -nx_out:]
    elif rank == 2:
        A_out[:] = A_in[-nx_out:, 0: nx_out]
    elif rank == 3:
        A_out[:] = A_in[-nx_out:, -nx_out:]


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
def smooth_sweep(b, a, h, shape):
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


def smooth(b, x, h, lplc, r, shape, iterations):

    for i in range(iterations):
        smooth_sweep(b, x, h, shape)

    laplacian(x, lplc, h)
    # Computing the residual.
    r[:] = b - lplc


def smooth_parallel(b, a, h, lplc, r, shape, iterations, com):
    """loop."""
    ny, nx = shape

    for i in range(iterations):
        com.fill_buffers()
        com.fill_var()
        smooth_sweep(b, a, h, shape)

    com.fill_buffers()
    com.fill_var()

    laplacian(a, lplc, h)
    # Computing the residual.
    r[:] = b - lplc

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


def gather_blocks(comm, M_block, M_full, rank):
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
        comm.Bcast(M_blocks[i], i)

    M_full[:nx2, :nx2] = M_blocks[0][:-1, :-1]
    M_full[:nx2, nx2:] = M_blocks[1][:-1, 2:]
    M_full[nx2:, :nx2] = M_blocks[2][2:, :-1]
    M_full[nx2:, nx2:] = M_blocks[3][2:, 2:]

    # if rank == 0:
    #     plt.pcolormesh(M_full)
    #     plt.show()


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
        h += [h0 * 2 ** (n + 1 - i)]

    x += [x0]
    b += [b0]
    lplc += [b0.copy()]
    r += [r0]
    h += [h0]

    buff = Buffers(x0.shape, x0, 1, 3)
    temp_coarse = np.zeros((2**(n-1)+2, (2**(n-1)+2)))

    err = epsilon + 1
    it = 0

    if True:
    # while err > epsilon:
        smooth_parallel(b0, x0, h0, lplc[-1], r0, x0.shape, 2, buff)
        coarse(r[-1], temp_coarse)
        # Reduce to b[2]
        gather_blocks(buff.comm, temp_coarse, b[-2], buff.rank)
        # Iterate until the residual is smaller than epsilon.
        for i in range(2, n+1):
            smooth(b[-i], x[-i], h[-i], lplc[-i], r[-i], x[-i].shape, 2)
            coarse(r[-i], b[-i - 1])

        smooth(b[0], x[0], h[0], lplc[0], r[0], x[0].shape, 3)
        for i in range(1, n):
            interpolate_add_to(x[i - 1], x[i])
            smooth(b[i], x[i], h[i], lplc[i], r[i], x[i].shape, 3)

        split(buff.rank, x[-2], temp_coarse)
        interpolate_add_to(temp_coarse, x[-1])
        smooth_parallel(b0, x0, h0, lplc[-1], r0, x0.shape, 3, buff)
        err = np.amax(np.abs(r[-1][1:-1, 1:-1]))
        it += 1
    print("v cycles :", it)


def test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n = 6
    epsilon = 0.1
    nx1 = 2 ** n + 1
    nx0 = 2 ** (n + 1) + 1
    nx1_sub = nx1 + 2

    print(f"nx = {nx1_sub}")
    x = np.linspace(-10, 10, nx0)
    y = np.linspace(-10, 10, nx0)
    h = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    r = X ** 2 + Y ** 2
    b0 = 50 * np.exp(-r * 4)
    r = (4 + X) ** 2 + Y ** 2
    b0 -= 50 * np.exp(-r * 4)
    r = (5 + X) ** 2 + (5 + Y) ** 2
    b0 += 50 * np.exp(-r * 4)
    r = (X - 5) ** 2 + (Y - 2) ** 2
    b0 -= 50 * np.exp(-r * 4)
    r = (X - 5) ** 2 + (Y + 2) ** 2
    b0 -= 50 * np.exp(-r * 4)

    b = np.zeros((nx1_sub, nx1_sub))
    split(rank, b0, b)

    a = np.zeros_like(b)
    R = np.zeros_like(a)
    t0 = perf_counter()
    poisson_multigrid(b, a, R, h, epsilon, n)
    t1 = perf_counter()
    print(f"time multi 1    {t1-t0} s")

    fig, ax = plt.subplots(1, 2)
    ax0, ax1 = ax
    ax0.pcolormesh(a)
    a_max = np.amax(np.abs(a))
    cm = ax1.pcolormesh(R / a_max, cmap="bwr")
    fig.suptitle(f"{nx0}x{nx0} grid points , rank{rank}")
    ax0.set_title("Phi")
    ax1.set_title("Residual / max(Phi)")
    plt.colorbar(cm)
    plt.show()

if __name__ == "__main__":
    test()
