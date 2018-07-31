#!/usr/bin/env python
"""
Topology of the 12-6 Lennard-Jones force field truncated at 2.5 sigma.
"""

import inspect
import os
import sys

if len(sys.argv) != 4:
    msg = """
        Usage: %s lj_params.txt file.in out.chgcar 
        Summary
        -------
        Creates a potential energy topological map using the
        12-6 LJ force field. The LJ parameters must be specified
        in the lj_params.txt file. Though it need not be called
        \lj_params.txt\), it must have the format:

        sigma_1 epsilon_1
        sigma_2 epsilon_2
        ...
        sigma_N epsilon_N

        where sigma_i and epsilon_i are the LJ parameters
        between the voxel and species i. The order of the
        species must match the order that those species appear
        in file.in

        For example, if you
        are interested in the migration of Li in Si-O, then
        sigma_1 = sigma_{Li-Si} and sigma_2 = sigma_{Li-O},
        and similarly for epsilon_1 and epsilon_2. If your
        input file has N species, lj_params.txt must have
        N lines. You do not need LJ parameters for cross 
        interactions.
        """ % sys.argv[0]
    print >>sys.stderr, msg
    sys.exit(0)

import numpy as np
from numpy.linalg import norm
from ase.io import read, write
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler as SourceModule

lj_file = sys.argv[1]
infile = sys.argv[2]
chgcar = sys.argv[3]

# read in atoms
if infile.endswith('.vasp') or infile.endswith('.poscar'):
    atoms = read(infile, format='vasp')
else:
    atoms = read(infile)
natoms = np.int32(atoms.get_number_of_atoms())
#
# atom types
#
ntypes = np.int32(0)
d = {}
for key in atoms.get_chemical_symbols():
    if not d.has_key(key):
        d[key] = ntypes
        ntypes += 1
atom_types = np.array(tuple(d[key] for key in
                            atoms.get_chemical_symbols()),
                      dtype=np.int32)
#
# periodicity
#
pv = np.asarray(atoms.get_cell(), dtype=np.float32)
pv = np.reshape(pv, (pv.size,))
#
# number of divisions
#
resolution = 0.1
ndiv = np.asarray(np.ceil([norm(v)/resolution for v in atoms.get_cell()]), dtype=np.int32)
#
# scaled atom positions
#
spos = np.asarray(atoms.get_scaled_positions(), dtype=np.float32)
spos = np.reshape(spos, (spos.size,))
#
# set up grid
#
ngrid = np.int32(np.prod(ndiv))
grid = np.ndarray((ngrid,), dtype=np.float32)
grid.fill(0.0)
#
# read in LJ parameters
#
try:
    # from parameter list, one species per line
    # to species list, one parameter, e.g. sigma or epsilon or rcut, per line
    lj_params = np.loadtxt(lj_file, dtype=np.float32).transpose()
    # expand each parameter into a matrix, but since we are only concerned
    # with s1 = voxel-s1, s2 = voxel-s2, ..., sN = voxel-sN reactions, this becomes::
    #     sigma = [[  s1, 0.0 ] = [s1, 0.0, 0.0, s2]
    #              [ 0.0,  s2 ]]
    # and similarly for epsilon and rcut. The resulting parameters
    # is an appending of the three, i.e.
    #     lj_params = sigma + epsilon + rcut
    #               = [s1, 0.0, 0.0, s2, e1, 0.0, 0.0, e2, r1, 0.0, 0.0, r2]
    lj_params = np.reshape(np.array([np.diag(row) for row in lj_params]), (-1,))
except:
    print >>sys.stderr, "ERROR: Failed to read %s" % lj_file
    sys.exit(1)

# -----------------------

try:
    device_id = int(os.environ['CUDA_DEVICE'])
except KeyError:
    device_id = 0
device = cuda.Device(device_id)

DIM = 3
# number of threads that one block can handle
THREADS_PER_BLOCK = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
# number of blocks required to handle all atoms
N_BLOCKS = int(np.ceil(float(atoms.get_number_of_atoms())/\
               THREADS_PER_BLOCK))
# maximum number of threads per block
N_THREAD = int(np.ceil(float(atoms.get_number_of_atoms())/\
               N_BLOCKS))

path=os.path.dirname(os.path.realpath(inspect.getfile(inspect.currentframe())))
# ------------------------
# precompiled kernel is preferrable
# ------------------------
kernel = path + '/Lennard-Jones_BpV.cubin'
if os.path.isfile(kernel):
    kernel = cuda.module_from_file(kernel)
else:
    # ------------------------
    # just-in-time compilation
    # ------------------------
    with open(path + '/Lennard-Jones_BpV.cu') as ifs:
        kernel_text = ifs.read()
    kernel = SourceModule(kernel_text)

energy = kernel.get_function('Lennard_Jones_BpV')

shared_mem = DIM * N_THREAD * np.float32().nbytes
cuda_grid = (int(ngrid), N_BLOCKS)
cuda_blocks = (N_THREAD, 1, 1)

# ###########################
# DEBUGGING
# ###########################
print "<<<<< RUN >>>>>"
print "atoms:", atoms
print "natoms:", natoms
print "pv:", pv
print "resolution:", resolution
print "ndiv:", ndiv
print "atom_types:", atom_types
print "ntypes:", ntypes
print "ngrid:", ngrid
print "lj_params:", lj_params
print "<<<<< CUDA >>>>>"
print "device:", device_id
print "DIM:", DIM
print "THREADS_PER_BLOCK:", THREADS_PER_BLOCK
print "N_BLOCKS:", N_BLOCKS
print "N_THREAD:", N_THREAD
print "path:", path
print "shared_mem:", (shared_mem, shared_mem/DIM/np.float32().nbytes)
# ###########################

energy(cuda.InOut(grid), ngrid,
       cuda.In(pv),
       cuda.In(ndiv),
       cuda.In(spos), cuda.In(atom_types), natoms,
       cuda.In(lj_params), ntypes,
       block=cuda_blocks, grid=cuda_grid, shared=shared_mem)

# -----------------------

write(chgcar, atoms, format='vasp', direct=True, vasp5=True)
with open(chgcar, 'a', buffering=1048576) as ofs:
    ofs.write('\n')
    ofs.write(' '.join([str(n) for n in ndiv]) + '\n')
    count = 1
    for val in grid:
        ofs.write('{:.12f}'.format(float(val)))
        ofs.write(' ' if count % 5 else '\n')
        count += 1

