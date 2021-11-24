##
#
#
# @file
# @version 0.1
run:
	mpiexec --mca opal_warn_on_missing_libcuda 0 -n 4 python parallel_multigrid.py
# end
