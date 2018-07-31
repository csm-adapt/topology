#!/usr/bin/env python
"""
Synopsis
--------
Reads the VASP CHGCAR-formatted file object into a Map.

@author: Branden Kappes
@date: 2013 July 28
"""

from topopropy.map import MapBase
from ase.atoms import Atoms
from ase.data import chemical_symbols
from operator import mul
import numpy as np
import sys


# pylint: disable=R0912
# pylint: disable=R0914
# pylint: disable=R0915
def read_chgcar_into_map(ifs, obj):
    """
    Synopsis
    --------
    Reads CHGCAR-formatted data from the input filestream
    into a topopropy Map-like object.

    Parameters
    ----------
    :ifs: Input file as either a filename or a filestream (file object).
    :obj: Destination, which must be derived from MapBase.

    Returns
    -------
    None
    """
    if not isinstance(obj, MapBase):
        msg = 'Type mimatch while reading from {:s}.'.format(ifs.name)
        msg += ' Destination is not derived from ase_ext.map.Map.'
        raise TypeError(msg)
    # if IFS is not a filestream, open it
    filestream = isinstance(ifs, file)
    if not filestream:
        ifs = open(ifs, 'r')
    try:
        lineno = 0
        optlines = 0
        natoms = 0
        elements = None
        grid_block = False
        grid_size = 0
        index = 0
        pv = []                                     # pylint: disable=C0103
        for line in ifs:
            # since most lines will be in the regular grid,
            # place this first
            if grid_block:
                for word in line.split():
                    obj[index] = float(word)
                    index += 1
                if index == grid_size:
                    break
            # read, and ignore, comment
            elif lineno < 1+optlines:
                pass
            # scaling factor
            elif lineno < 2+optlines:
                scaling_factor = float(line)
            # periodicity vectors
            elif lineno < 5+optlines:
                pv.append([scaling_factor*float(w) for w in line.split()])
            # read number of each specie and elements (if present)
            elif lineno < 6+optlines:
                try:
                    species = [int(w) for w in line.split()]
                    obj.set_pv(pv)
                    natoms = sum(species)
                    atom_pos = np.zeros((natoms, 3))
                    index = 0
                except ValueError:
                    elements = line.split()
                    optlines += 1
            # selective dynamics/direct or cartesian coordinates
            elif lineno < 7+optlines:
                if line[0].lower() == 's':
                    # ignore selective dynamics line
                    optlines += 1
                else:
                    direct = (line[0].lower() == 'd')
            # read atom positions
            elif lineno < 8+optlines+(natoms-1):
                atom_pos[index, :] = [float(x) for x in line.split()[:3]]
                if direct:
                    atom_pos[index, :] = np.dot(atom_pos[index], pv)
                else:
                    atom_pos[index] *= scaling_factor
                index += 1
            # blank line (end of atom data)
            elif lineno < 9+optlines+(natoms-1):
                # no elements specified (old style)
                if elements is None:
                    # use atoms in sequence (H, He, Li, ...)
                    elements = chemical_symbols[1:(len(species)+1)]
                # replicate symbols so the list of symbols is the
                # same length as the atom positions
                elements = sum([i*[e]
                                for i, e in zip(species, elements)], [])
                # store atom information
                obj.atoms = Atoms(
                    symbols=elements,
                    positions=atom_pos,
                    cell=pv)
            # read grid size
            elif lineno < 10+optlines+(natoms-1):
                nbins = [int(x) for x in line.split()]
                obj.set_ndiv(nbins)
                grid_size = reduce(mul, nbins)
                grid_block = True
                index = 0
            lineno += 1
    except:
        msg = 'ERROR: An error occured while reading {:s} ' \
              'into a topopropy Map'.format(ifs.name)
        print >> sys.stderr, msg
        raise
    # close file, if opened
    if not filestream:
        ifs.close()
#end 'def read_chgcar_into_map(ifs, obj):'
