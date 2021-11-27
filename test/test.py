#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from mpi4py import MPI
from parallel_multigrid import (Multigrid, gather_blocks,
                                split, residual)

def print_results(multigrid, t_point, t_it, rel_err):
    print_results.counter = vars(print_results).setdefault('counter', 0)
    print_results.t_old = vars(print_results).setdefault('t_old', 0)
    print_results.t_best = vars(print_results).setdefault('t_best', 1)
    if t_point < print_results.t_old:
        better = "+"
        if print_results.t_best > t_point:
            better += "+"
            print_results.t_best = t_point
    else:
        better = "-"

    if print_results.counter == 0:
        result_col = (f"{'n':<2}{'n_para':>8}{'max_rel_er':>12}"
                      f"{'t_per_point':>14}{'t_solve':>11}{'v_cycles':>10}"
                      f"{'improv':>8}")
        print(result_col)
        print_results.counter += 1

    val = (f"{multigrid.n:<2}{multigrid.n_para:>8}{rel_err:>12.1e}"
           f"{t_point:>14.2e}{t_it:>11.2e}{multigrid.it:>10}{better:>8}")
    print(val)
    print_results.t_old = t_point

def test(n, n_para):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    b_max = 50
    epsilon = b_max*0.0005

    # Domain size
    nx1 = 2 ** n + 1
    nx0 = 2 ** (n + 1) + 1
    x = np.linspace(-10, 10, nx0)
    y = np.linspace(-10, 10, nx0)
    # Step
    h = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # Right hand side
    # Create some vortex in the whole domain
    r = np.zeros_like(X)
    b0 = np.zeros_like(X)
    sign = 1
    xr = [ 4, -2,  6,  3,  3,  5, -5, -7]
    yr = [-2,  6,  6, -5,  5, -5, -2, -2]
    for x, y in zip(xr, yr):
        r = (X-x) ** 2 + (Y-y) ** 2
        b0 += sign*b_max * np.exp(-r * 7)
        sign *= -1

    # Attribute the subdomains to the processors
    # by spliting the whole domain.
    b = np.zeros((nx1+1, nx1+1))
    split(b0, b, rank)

    # a is the unknow
    a = np.zeros_like(b)
    # R is the residual
    R = np.zeros_like(a)

    # initialize the solver
    poisson = Multigrid(b, a, R, h, epsilon, n, n_para)

    # solve
    t = 0
    it = 10
    for i in range(it):
        b[1:-1] += b[0:-2]
        b[:] /= 2
        t0 = perf_counter()
        err = poisson.solve()
        t1 = perf_counter()
        t += t1-t0
    m_err = comm.reduce(err, MPI.MAX, 0)

    # Print results
    if rank == 0:
        print_results(poisson, t/(it*2**(n+1)-1), t/it, m_err/b_max)


if __name__ == "__main__":

    for n in range(4, 9):
        for n_para in range(n//3, n):
            test(n, n_para)

    # a_full = np.zeros_like(b0)
    # R_full = np.zeros_like(b0)
    # gather_blocks(comm, a, a_full)
    # gather_blocks(comm, R, R_full)
    # residual(R_full, a_full, b0, h)

    # if rank == 2:
    #     fig, ax = plt.subplots(1, 2)
    #     ax0, ax1 = ax
    #     ax0.pcolormesh(a_full)
    #     r_max = np.amax(np.abs(R_full/b_max))
    #     cm = ax1.pcolormesh(R_full / b_max, cmap="bwr", vmin=-r_max, vmax=r_max)
    #     fig.suptitle(f"{nx0}x{nx0} grid points")
    #     ax0.set_title("phi")
    #     ax1.set_title(f"Residual / max(B) max={r_max:.1e}")
    #     ax1.set_aspect('equal')
    #     ax0.set_aspect('equal')
    #     ax0.axis('off')
    #     ax1.axis('off')
    #     # plt.colorbar(cm)
    #     plt.savefig("test.png")
    #     plt.show()
