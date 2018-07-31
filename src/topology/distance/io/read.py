#!/usr/bin/env python
"""
Synopsis
--------
Entry point for all topology.distance read operations

@author: Branden Kappes
@date: 2013 July 28
"""

from topology.distance.grid import StructuredGrid
from topology.distance.map import MapBase
from topology.distance.io.guess_format import guess_format
from topology.distance.io.read_chgcar_into_grid import read_chgcar_into_grid
from topology.distance.io.read_chgcar_into_map import read_chgcar_into_map


def read(ifs, obj, fmt=None):
    """
    Synopsis
    --------
    Reads the object in either the specified format
    (if given) or attempts to guess the format from
    IFS.

    Parameters
    ----------
    :ifs (file): input file object
    :obj (StructuredGrid-like): object to hold the result (NOTE:
        will be overwritten!)
    :format (string): specify the output file format
                   {{CHGCAR|chgcar}|}

    Returns
    -------
    None
    """
    # try to guess the filetype
    if fmt is None:
        fmt = guess_format(ifs)
    # is IFS a filestream or filename?
    filestream = isinstance(ifs, file)
    if not filestream:
        ifs = open(ifs, 'r')
    # read CHGCAR based on the type of object
    if fmt.lower() == 'chgcar':
        if isinstance(obj, MapBase):
            read_chgcar_into_map(ifs, obj)
        elif isinstance(obj, StructuredGrid):
            read_chgcar_into_grid(ifs, obj)
        else:
            TypeError('Unrecognized grid-like object.')
    if not filestream:
        ifs.close()
#end 'def write_chgcar(ofs, obj):'
