# Multigrid
 **Parallel multigrid solver for Poisson equation.**
 - The fine grid levels are computed in parallel via simple Jacobi iterations.
 - The coarse grid levels are computed serially to reduce the overhead of parallel exchanges, using weighted Gauss-Seidel iterations.
 - For now the domain geometry (square) and the number of processors (4) is hardcoded.

![alt text](visual.gif)
