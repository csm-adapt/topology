#!/usr/bin/env python
"""
Synopsis
--------
Reads a CHGCAR-formatted file into a StructuredGrid object.

@author: Branden Kappes
@date: 2013 July 28
"""

from topopropy.grid import StructuredGrid
from operator import mul
import sys


# reads VASP CHGCAR-formatted filestream.
# pylint: disable=R0912
# pylint: disable=R0914
# pylint: disable=R0915
def read_chgcar_into_grid(ifs, obj):
    """
    Reads CHGCAR-formatted data from the input filestream
    into the topopropy.StructuredGrid object.
    """
    if not isinstance(obj, StructuredGrid):
        msg = 'Type mimatch while reading from {:s}.'.format(ifs.name)
        msg += ' Destination is not derived from ase_ext.grid.StructuredGrid.'
        raise TypeError(msg)
    # if IFS is not a filestream, open it
    filestream = isinstance(ifs, file)
    if not filestream:
        ifs = open(ifs, 'r')
    try:
        lineno = 0
        optlines = 0
        natoms = 0
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
                    index = 0
                # ignore element information
                except ValueError:
                    optlines += 1
            # selective dynamics/direct or cartesian coordinates
            elif lineno < 7+optlines:
                if line[0].lower() == 's':
                    # ignore selective dynamics line
                    optlines += 1
                else:
                    # since StructuredGrid has no atom information,
                    # whether the atoms are in real or scaled coordinates
                    # is irrelevant
                    #direct = (line[0].lower() == 'd')
                    pass
            # read, and ignore, atom positions
            elif lineno < 9+optlines+(natoms-1):
                pass
            # read grid size
            elif lineno < 10+optlines+(natoms-1):
                nbins = [int(x) for x in line.split()]
                obj.set_ndiv(nbins)
                grid_size = reduce(mul, nbins)
                grid_block = True
                index = 0
            lineno += 1
    except:
        msg = 'ERROR: An error occured while reading {:s} '\
                'into a topopropy StructuredGrid'.format(ifs.name)
        print >> sys.stderr, msg
        raise
    # close file, if opened
    if not filestream:
        ifs.close()
#end 'def read_chgcar_into_map(ifs, obj):'
