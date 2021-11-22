#!/usr/bin/env python3

"""Multigrid solver for Poisson equation."""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
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
def smooth(b, a, h, lplc, r, shape, iterations):
    """Gauss-Seidel method for multigrid."""
    ny, nx = shape

    for it in range(iterations):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                a[j, i] = 0.25 * (
                    a[j, i + 1] + a[j, i - 1] + a[j + 1, i] +
                    a[j - 1, i] - h**2*b[j, i])
        # Go back in the opposite direction
        for j in range(ny - 2, 0, -1):
            for i in range(nx - 2, 0, -1):
                a[j, i] = 0.25 * (
                    a[j, i + 1] + a[j, i - 1] + a[j + 1, i] +
                    a[j - 1, i] - h**2*b[j, i])

    laplacian(a, lplc, h)
    # Computing the residual.
    r[:] = b - lplc


@jit(nopython=True, cache=True)
def poisson_checkerboard(b, a, k, shape):
    """Alternate iterations in a checker-board patern."""
    ny = shape[0]
    nx = shape[1]
    for j in range(1, ny-1):
        k %= 2
        for ix in range(nx // 2 + nx % 2 * (1-k)-1):
            i = ix*2 + k + 1
            a[j, i] = 0.25*(a[j, i+1]+a[j, i-1] +
                            a[j+1, i] + a[j-1, i] + b[j, i])
        k += 1

@jit(nopython=True, cache=True)
def gauss_seidel(b, a, h, lplc, r, shape, epsilon):
    """Gauss-Seidel method for comparaison."""
    ny, nx = shape
    max_residual = epsilon + 1

    while max_residual > epsilon:
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                a[j, i] = 0.25 * (
                    a[j, i + 1] + a[j, i - 1] + a[j + 1, i] +
                    a[j - 1, i] - h**2*b[j, i])
        # Go back in the opposite direction
        for j in range(ny - 2, 0, -1):
            for i in range(nx - 2, 0, -1):
                a[j, i] = 0.25 * (
                    a[j, i + 1] + a[j, i - 1] + a[j + 1, i] +
                    a[j - 1, i] - h**2*b[j, i])

        laplacian(a, lplc, h)
        max_residual = np.amax(np.abs(b[1:-1, 1:-1]-lplc[1:-1, 1:-1]))
    # Computing the residual.
    r[:] = b - lplc


@jit(nopython=True, cache=True)
def coarse(a, a_crs):
    """Reduction on coarser grid."""
    # a_left = a[0, 0] / 8 + a[1, 0] / 16
    # for i in range(1, a_crs.shape[1] - 1):
    #     a_right = a[0, 2 * i + 1] / 8 + a[1, 2 * i + 1] / 16
    #     a_crs[0, i] = a[0, 2 * i] / 2 + a[1, 2 * i] / 8
    #     a_crs[0, i] += a_right + a_left
    #     a_left = a_right

    for j in range(1, a_crs.shape[0] - 1):
        a_left = a[2 * j, 1] / 8 + (a[2 * j + 1, 1] + a[2 * j - 1, 1]) / 16
        # a_crs[j, 0] = a_left + a[2 * j, 0] / 2 + (a[2 * j + 1, 0] + a[2 * j - 1, 0]) / 8
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
        # a_crs[j, -1] = (
        #     a_left + a[2 * j, -1] / 2 + (a[2 * j + 1, -1] + a[2 * j - 1, -1]) / 8
        # )

    # a_left = a[-1, 0] / 8 + a[-2, 0] / 16
    # for i in range(1, a_crs.shape[1] - 1):
    #     a_right = a[-1, 2 * i + 1] / 8 + a[-2, 2 * i + 1] / 16
    #     a_crs[-1, i] = a[-1, 2 * i] / 2 + a[-2, 2 * i] / 8
    #     a_crs[-1, i] += a_right + a_left
    #     a_left = a_right


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


def poisson_multigrid(b, a0, r0, h0, epsilon, n):
    """
    Solve the Poisson equation by reapeating v cycles.
    """
    ny, nx = a0.shape
    a = []
    bs = []
    lplc = []
    r = []
    h = []
    for i in range(1, n):
        a += [np.zeros((2 ** i + 1, 2 ** i + 1))]
        bs += [np.zeros((2 ** i + 1, 2 ** i + 1))]
        lplc += [np.zeros((2 ** i + 1, 2 ** i + 1))]
        r += [np.zeros((2 ** i + 1, 2 ** i + 1))]
        h += [h0*2**(n-i)]

    a += [a0]
    bs += [b]
    lplc += [b.copy()]
    r += [r0]
    h += [h0]
    err = epsilon + 1
    it = 0

    while err > epsilon:
        # Iterate until the residual is smaller than epsilon.
        for i in range(1, n):
            smooth(bs[-i], a[-i], h[-i], lplc[-i], r[-i], a[-i].shape, 3)
            coarse(r[-i], bs[-i-1])

        smooth(bs[0], a[0], h[0], lplc[0], r[0], a[0].shape, 10)
        for i in range(1, n):
            interpolate_into(a[i-1], a[i])
            smooth(bs[i], a[i], h[i], lplc[i], r[i], a[i].shape, 20)

        err = np.amax(np.abs(r[-1][1:-1, 1:-1]))
        it += 1
    print("v cycles :", it)


def poisson(b, a, r, h, epsilon):
    """
    Solve the Poisson equation using Gauss-Seidel Method.
    """
    lplc = b.copy()

    gauss_seidel(b, a, h, lplc, r, a.shape, epsilon)


def test():
    n = 8
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
    b = b0*1.001

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
    cm = ax1.pcolormesh(R/a_max, cmap="bwr")

    fig.suptitle("1024x1024 grid points")
    ax0.set_title("Phi")
    ax1.set_title("Residual / max(Phi)")
    plt.colorbar(cm)
    plt.show()

    # a[:] = 0
    # t0 = perf_counter()
    # poisson(b0, a, R, h, epsilon)
    # t1 = perf_counter()
    # print(f"time poisson 1  {t1-t0} s", flush=True)
    # t0 = perf_counter()
    # poisson(b, a, R, h, epsilon)
    # t1 = perf_counter()
    # print(f"time poisson 2  {t1-t0} s", flush=True)
    # plt.pcolormesh(a)
    # plt.colorbar()
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
