#!/usr/bin/env python
"""
Synopsis
--------
Frontend for all output operations.

@author: Branden Kappes
@date: 2013 July 28
"""

from topology.distance.io.guess_format import guess_format
from topology.distance.map import MapBase
from topology.distance.io.write_map_to_chgcar import write_map_to_chgcar


def write(ofs, obj, fmt=None):
    """
    Synopsis
    --------
    Writes the object in either the specified format
    (if given) or attempts to guess the format from
    OFS.

    Parameters
    ----------
    :ofs (file): output file stream
    :obj (Map-like): object to write
    :format (string, optional): specify the output file format
        {{CHGCAR|chgcar}|}

    Returns
    -------
    None
    """
    # try to guess the filetype
    if fmt is None:
        fmt = guess_format(ofs)
    # is OFS a filestream or filename?
    filestream = isinstance(ofs, file)
    if not filestream:
        ofs = open(ofs, 'w')
    # write CHGCAR based on the type of object
    if fmt.lower() == 'chgcar':
        if isinstance(obj, MapBase):
            write_map_to_chgcar(ofs, obj)
        else:
            TypeError('Unrecognized grid-like object.')
    if not filestream:
        ofs.close()
#end 'def write_chgcar(ofs, obj):'
