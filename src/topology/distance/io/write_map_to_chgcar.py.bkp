#!/usr/bin/env python
"""
Synopsis
--------
Writes a :code:`topopropy.Map` object to a file in the
VASP CHGCAR format.

@author: Branden Kappes
@date: 2013 July 28
"""

import numpy as np


def write_map_to_chgcar(ofs, dmap):
    """
    Synopsis
    --------
    Writes the data stored in the Map-like object to *ofs* in the
    VASP CHGCAR format.

    Parameters
    ----------
    :ofs (file): Ouptut file stream, *i.e.* an opened file.
    :dmap (Map-like): The Map-like object that holds the structured grid.

    Returns
    -------
    None
    """
    points = dmap.atoms.positions
    nbins = dmap.ndiv
    pv = dmap.pv                                    # pylint: disable=C0103
    ipv = dmap.ipv
    # element information and number of each element
    elements = tuple(set(dmap.get_chemical_symbols()))
    nelements = {}
    ielements = {}
    for atom in dmap.atoms:
        index = atom.index
        symbol = atom.symbol
        if symbol not in nelements:
            nelements[symbol] = 1
            ielements[symbol] = [index]
        else:
            nelements[symbol] += 1
            ielements[symbol].append(index)
    # write the CHGCAR
    # ... comment line
    print >> ofs, dmap.atoms.get_chemical_formula()
    # ... scaling factor
    print >> ofs, '  %.9f' % (1.,)
    # ... periodicity matrix
    print >> ofs, '% 12.9f % 12.9f % 12.9f' % tuple(pv[0])
    print >> ofs, '% 12.9f % 12.9f % 12.9f' % tuple(pv[1])
    print >> ofs, '% 12.9f % 12.9f % 12.9f' % tuple(pv[2])
    # ... chemical symbols of all elements. This sets the order in which
    # ... the atom information will be written
    print >> ofs, ' ' + ' '.join(elements)
    # ... number of each atom species, same order as above
    print >> ofs, ' ' + ' '.join(['{:d}'.format(nelements[symbol])
                                 for symbol in elements])
    # ... direct (scaled) coordinates
    print >> ofs, 'Direct'
    # ... write the atoms in order of their element
    for symbol in elements:
        for i in ielements[symbol]:
            sx = np.dot(points[i], ipv)             # pylint: disable=C0103
            print >> ofs, '% 12.9f % 12.9f % 12.9f' % tuple(sx)
    print >> ofs, ''
    print >> ofs, '%d %d %d' % tuple(nbins)
    i = 0
    try:
        for voxel in dmap:
            i += 1
            ofs.write('{:.9f} '.format(float(voxel)))
            if i % 5 == 0:
                ofs.write('\n')
        if i % 5 != 0:
            ofs.write('\n')
    except ValueError, e:
        print "Offending value:", voxel
        raise e
#end 'def write_map_to_chgcar(ofs, dmap):'

