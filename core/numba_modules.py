#!/usr/bin/env python3
import numpy as np
from numba.pycc import CC

cc = CC("multigrid_module")
# Uncomment the following line to print out the compilation steps
# cc.verbose = True


@cc.export("split", "(double[:, :], double[:, :], int64)")
def split(A_in, A_out, rank):
    nx_out, ny = A_out.shape
    if rank == 0:
        A_out[:] = A_in[0:nx_out, 0:nx_out]
    elif rank == 1:
        A_out[:] = A_in[0:nx_out, -nx_out:]
    elif rank == 2:
        A_out[:] = A_in[-nx_out:, 0:nx_out]
    elif rank == 3:
        A_out[:] = A_in[-nx_out:, -nx_out:]


@cc.export("laplacian", "(f8[:, :], f8[:, :], f8)")
def laplacian(a, lplc, h=1):
    """Compute the laplacian of a."""
    ny, nx = a.shape
    h = 1 / (h ** 2)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            lplc[j, i] = (
                a[j, i - 1] + a[j, i + 1] + a[j - 1, i] + a[j + 1, i] - 4 * a[j, i]
            ) * h


@cc.export("residual", "(f8[:,::1], f8[:, ::1], f8[:, ::1], f8)")
def residual(r, a, b, h):
    """Compute the residual of a."""
    ny, nx = a.shape
    h = 1 / (h ** 2)
    for j in range(1, ny - 1):
        left = a[j, 0] * h
        for i in range(1, nx - 1):
            cent = a[j, i] * h
            r[j, i] = (
                b[j, i]
                - left
                - (a[j, i + 1] + a[j - 1, i] + a[j + 1, i]) * h
                + cent * 4
            )
            left = cent


@cc.export("smooth_sweep_jacobi", "(f8[:, ::1], f8[:, ::1], f8[:, ::1], f8)")
def smooth_sweep_jacobi(b, a, a_temp, h):
    """Gauss-Seidel method for multigrid."""
    ny, nx = a.shape
    h = h ** 2

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            a_temp[j, i] = 0.25 * (
                a[j, i + 1] + a[j, i - 1] + a[j + 1, i] + a[j - 1, i] - h * b[j, i]
            )
    for j in range(ny):
        for i in range(nx):
            a[j, i] = a_temp[j, i]


@cc.export("smooth", "(f8[:, ::1], f8[:, ::1], f8, f8[:, ::1], int64)")
def smooth(b, x, h, r, iterations):
    a = 1.5
    alpha = 1 - a
    beta = a / 4
    h = h ** 2
    ny, nx = x.shape
    for k in range(iterations):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                x[j, i] = alpha * x[j, i] + beta * (
                    x[j, i + 1] + x[j, i - 1] + x[j + 1, i] + x[j - 1, i] - h * b[j, i]
                )

    h = 1 / h
    # Compute the residual
    for j in range(1, ny - 1):
        left = x[j, 0] * h
        for i in range(1, nx - 1):
            cent = x[j, i] * h
            r[j, i] = (
                b[j, i]
                - left
                - (x[j, i + 1] + x[j - 1, i] + x[j + 1, i]) * h
                + cent * 4
            )
            left = cent


@cc.export("smooth_altern", "(f8[:, ::1], f8[:, ::1], f8, f8[:, ::1], int64)")
def smooth_altern(b, x, h, r, iterations):
    a = 1.5
    alpha = 1 - a
    beta = a / 4
    h = h ** 2
    ny, nx = x.shape
    for k in range(iterations):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                x[j, i] = alpha * x[j, i] + beta * (
                    x[j, i + 1] + x[j, i - 1] + x[j + 1, i] + x[j - 1, i] - h * b[j, i]
                )
        for j in range(ny - 2, 0, -1):
            for i in range(nx - 2, 0, -1):
                x[j, i] = alpha * x[j, i] + beta * (
                    x[j, i + 1] + x[j, i - 1] + x[j + 1, i] + x[j - 1, i] - h * b[j, i]
                )

    h = 1 / h
    # Compute the residual
    for j in range(1, ny - 1):
        left = x[j, 0] * h
        for i in range(1, nx - 1):
            cent = x[j, i] * h
            r[j, i] = (
                b[j, i]
                - left
                - (x[j, i + 1] + x[j - 1, i] + x[j + 1, i]) * h
                + cent * 4
            )
            left = cent


@cc.export("coarse", "(f8[:, :], f8[:, :], int64, int64)")
def coarse(a, a_crs, ofst_i=0, ofst_j=0):
    """Reduction on coarser grid."""
    for j in range(1, a_crs.shape[0] - 1):
        a_left = (
            a[2 * j + ofst_j, 1 + ofst_i] / 8
            + (a[2 * j + ofst_j + 1, 1 + ofst_i] + a[2 * j + ofst_j - 1, 1 + ofst_i])
            / 16
        )
        for i in range(1, a_crs.shape[1] - 1):
            a_right = (
                a[2 * j + ofst_j, 2 * i + ofst_i + 1] / 8
                + (
                    a[2 * j + ofst_j + 1, 2 * i + ofst_i + 1]
                    + a[2 * j + ofst_j - 1, 2 * i + ofst_i + 1]
                )
                / 16
            )
            a_crs[j, i] = (
                a[2 * j + ofst_j, 2 * i + ofst_i] / 4
                + (
                    a[2 * j + ofst_j + 1, 2 * i + ofst_i]
                    + a[2 * j + ofst_j - 1, 2 * i + ofst_i]
                )
                * 1
                / 8
            )
            a_crs[j, i] += a_right + a_left
            a_left = a_right


@cc.export("interpolate_add_to", "(f8[:, :], f8[:, :], int64, int64)")
def interpolate_add_to(a, a_new, ofst_i=0, ofst_j=0):
    """Interpolate."""
    for j in range(1, a.shape[0] - 1):
        for i in range(1, a.shape[1] - 1):
            a_new[2 * j + ofst_j, 2 * i + ofst_i] += a[j, i]
    for j in range(0, a.shape[0] - 1):
        for i in range(0, a.shape[1] - 1):
            a_new[2 * j + 1 + ofst_j, 2 * i + 1 + ofst_i] += (
                a[j + 1, i + 1] + a[j + 1, i] + a[j, i + 1] + a[j, i]
            ) / 4
    for j in range(1, a.shape[0] - 1):
        for i in range(0, a.shape[1] - 1):
            a_new[2 * j + ofst_j, 2 * i + 1 + ofst_i] += (a[j, i] + a[j, i + 1]) / 2

    for j in range(0, a.shape[0] - 1):
        for i in range(1, a.shape[1] - 1):
            a_new[2 * j + 1 + ofst_j, 2 * i + ofst_i] += (a[j, i] + a[j + 1, i]) / 2


if __name__ == "__main__":
    cc.compile()
