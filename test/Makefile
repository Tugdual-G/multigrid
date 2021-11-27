##
#
#
# @file
# @version 0.1
# mpiexec -n 4 python -m cProfile -o program.prof parallel_multigrid.py
#tuna program.prof
run:
	mpiexec --mca opal_warn_on_missing_libcuda 0 -n 4 python test.py
# end
