#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from numba import jit


@jit(nopython=True, cache=True)
def smooth_damp(b, x, h, r, iterations, a=1):
    ny, nx = x.shape
    for k in range(iterations):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                x[j, i] = (1-a)*x[j, i]+a*0.25 * (
                    x[j, i + 1]
                    + x[j, i - 1]
                    + x[j + 1, i]
                    + x[j - 1, i]
                    - h ** 2 * b[j, i]
                )
    # Compute the residual
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            r[j, i] = b[j, i]-(
                x[j, i - 1] + x[j, i + 1] + x[j - 1, i] +
                x[j + 1, i] - 4 * x[j, i]
            ) / (h ** 2)


n = 6
b_max = 50
epsilon = b_max*0.001
nx0 = 2 ** n + 1
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

a = np.zeros_like(b0)
R = np.zeros_like(a)

for i in range(10):
    t0 = perf_counter()
    smooth_damp(b0, a, h, R, 100, 1.5)
    t1 = perf_counter()
    err = np.max(np.abs(R))
    print(err)
