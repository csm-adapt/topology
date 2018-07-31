#!/usr/bin/env python
"""
Synopsis
--------
Attempts to guess the format given a filename/file object.
"""

import os


def guess_format(filename):
    """
    Synopsis
    --------
    Attempts to guess the format of *filename*
    from its extension, if present.

    Parameters
    ----------
    :filename (string): The name of the file whose format is desired.
    """
    if isinstance(filename, file):
        filename = filename.name
    fullpath = os.path.normpath(filename)
    basename = os.path.split(fullpath)[1]
    basename, ext = os.path.splitext(basename)
    # first, check extensions for known formats
    if ext.lower() == '.chgcar':
        return 'chgcar'
    # then, check basename for known formats
    elif basename in ('CHG', 'CHGCAR'):
        return 'chgcar'
    # FUTURE: open the file and perform a perfunctory scan to
    # determine the format.
    else:
        raise ValueError('Unrecognized output file format')
#end 'def _get_format_from_extension(filename):'
