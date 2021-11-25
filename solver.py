#!/usr/bin/env python3

"""Multigrid solver for Poisson equation."""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, stencil
from time import perf_counter


@jit(nopython=True, cache=True)
def laplacian(a, lplc, h=1):
    """Compute the laplacian of a."""
    ny, nx = a.shape
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            lplc[j, i] = (a[j, i-1] + a[j, i+1] + a[j-1, i] +
                          a[j+1, i] - 4*a[j, i])/(h**2)


@jit(nopython=True, cache=True)
def gauss_seidel(b, a, h, lplc, r, shape, iterations):
    """Gauss-Seidel method for comparaison."""
    ny, nx = shape
    a_new = np.zeros_like(a)

    for k in range(iterations):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                a[j, i] = 0.25 * (
                    a[j, i + 1] + a[j, i - 1] + a[j + 1, i] +
                    a[j - 1, i] - h**2*b[j, i])
        # # Go back in the opposite direction
        # for j in range(ny - 2, 0, -1):
        #     for i in range(nx - 2, 0, -1):
        #         a[j, i] = 0.25 * (
        #             a[j, i + 1] + a[j, i - 1] + a[j + 1, i] +
        #             a[j - 1, i] - h**2*b[j, i])

    laplacian(a, lplc, h)
    # Computing the residual.
    r[:] = b - lplc
@stencil
def jacob(a, h):
    return 0.25*(a[0, 0, 1] + a[0, 0, -1] + a[0, -1, 0] + a[0, 1, 0]- h**2*a[1, 0, 0])

@jit(nopython=True, cache=True)
def gauss_seidel_diag(b, a, h, lplc, r, shape, iterations):
    """Gauss-Seidel method for comparaison."""
    ny, nx = shape
    a_new = np.zeros((2, nx, nx))
    a_new[1] = b
    a_new[0] = a

    for k in range(iterations):
        jacob(a_new, h)
        # # Go back in the opposite direction
        # for j in range(ny - 2, 0, -1):
        #     for i in range(nx - 2, 0, -1):
        #         a[j, i] = 0.25 * (
        #             a[j, i + 1] + a[j, i - 1] + a[j + 1, i] +
        #             a[j - 1, i] - h**2*b[j, i])

    a[:] = a_new[0, :, :]
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            lplc[j, i] = (a[j, i-1] + a[j, i+1] + a[j-1, i] +
                          a[j+1, i] - 4*a[j, i])/(h**2)
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
def interpolate_into(a, a_add):
    """Interpolate."""
    a_i = np.zeros_like(a_add)
    for j in range(1, a.shape[0]-1):
        for i in range(1, a.shape[1]-1):
            a_i[2 * j, 2 * i] = a[j, i]
    for j in range(0, a.shape[0]-1):
        for i in range(0, a.shape[1]-1):
            a_i[2 * j + 1, 2 * i + 1] = (
                a[j + 1, i + 1]
                + a[j + 1, i]
                + a[j, i + 1]
                + a[j, i]
            ) / 4
    for j in range(1, a.shape[0]-1):
        for i in range(0, a.shape[1]-1):
            a_i[2*j, 2*i+1] = (a[j, i] + a[j, i+1]) / 2

    for j in range(0, a.shape[0]-1):
        for i in range(1, a.shape[1]-1):
            a_i[2*j+1, 2*i] = (a[j, i] + a[j+1, i]) / 2

    # a_add[1:-1, 1:-1] += a_i[1:-1, 1:-1]
    a_add += a_i



def test():
    n = 5
    epsilon = 0.001
    nx = 2 ** n + 1
    print(f"nx = {nx}")
    ny = nx
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    h = y[1]-y[0]
    X, Y = np.meshgrid(x, y)
    r = X ** 2 + Y ** 2
    b = 50*np.exp(-r*4)
    r = (4+X) ** 2 + Y ** 2
    b -= 50*np.exp(-r*4)
    r = (5+X) ** 2 + (5+Y) ** 2
    b += 50*np.exp(-r*4)
    r = (X-5) ** 2 + (Y-2) ** 2
    b -= 50*np.exp(-r*4)
    a = np.zeros_like(b)
    b0 = b.copy()
    b = b0*1.01

    R = np.zeros_like(a)
    lplc = np.zeros_like(a)
    a = np.zeros_like(b)
    gauss_seidel_diag(b0, a, h, lplc, R, (nx, nx), 100)

    t0 = perf_counter()
    for i in range(50):
        a = np.zeros_like(b)
        gauss_seidel(b0, a, h, lplc, R, (nx, nx), 100)
    t1 = perf_counter()
    max_r = np.amax(np.abs(R))
    print(f"time multi 1    {t1-t0} s, max r {max_r}", flush=True)

    a = np.zeros_like(b)
    t0 = perf_counter()
    for i in range(50):
        a = np.zeros_like(b)
        gauss_seidel_diag(b0, a, h, lplc, R, (nx, nx), 100)
    t1 = perf_counter()
    max_r = np.amax(np.abs(R))
    print(f"time multi 2    {t1-t0} s, max r {max_r}", flush=True)


    # fig, ax = plt.subplots(1, 2)
    # ax0, ax1 = ax
    # ax0.pcolormesh(a)
    # a_max = np.amax(np.abs(a))
    # cm = ax1.pcolormesh(R/a_max, cmap="bwr")

    # fig.suptitle("1024x1024 grid points")
    # ax0.set_title("Phi")
    # ax1.set_title("Residual / max(Phi)")
    # plt.colorbar(cm)
    # plt.show()



def test_interpol():
    from matplotlib.image import imread

    n = 7
    nx = 2 ** n + 1
    img = imread("image.jpg")[::-1, ::-1, 1]
    A = img[20:20+nx, 80:80+nx]
    Ac1 = np.zeros((2**(n-1)+1, 2**(n-1)+1))
    Ac2 = np.zeros((2**(n-2)+1, 2**(n-2)+1))
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

if __name__=="__main__":

    test()
