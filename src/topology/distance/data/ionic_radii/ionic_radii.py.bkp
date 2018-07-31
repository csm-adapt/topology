#!/usr/bin/env python
"""
Returns the ionic radii for an atom in a given charge and
coordination state. The use and meaning of ionic radius is
a matter of some discussion, as measuring this physical
property is rather tenuous.

These data are taken from
http://abulafia.mt.ic.ac.uk/shannon/radius.php?orderby=Ion&dir=1
accessed 20 Mar 2013. In turn, these were tabulated from
R.D. Shannon, "Revised Effective Ionic Radii and Systematic Studies of
Interatomic Distances in Halides and Chalcogenides", Acta
Crystallographica A32 (1976) 751-767.
"""


def static_data(varname, filename):
    """
    Generator that creates a decorator to store and make
    accessible static data.
    """
    import json
    import os
    fullpath = os.path.join(os.path.dirname(__file__), filename)

    def decorate(func):
        """Function decorator."""
        with open(fullpath, 'r') as ifs:
            setattr(func, varname, json.load(ifs))
        return func
    return decorate


@static_data('data', 'ionic_radii.json')
def ionic_radii(symbol=None,
                charge=None,
                coordination=None,
                spin=None):
    """
    Synopsis
    --------
    As the ionic radius will change with the conditions under which
    an atom is exposed, a simple list is only an approximation of
    the actual ionic radii; and even then, only for those species
    with a "normal" charge state.

    An explanation is in order for the proper usage of this function::

        ionic_radii(): returns the entire dataset as a list of tuples
                       [(element, charge, coordination, spin,
                               crystal radius, ionic radius),...]

    any other combination of parameters includes only matching parameters.
    So if, for example, charge=3 , then the resulting list would be::

        [(element0, 3, coordination0, spin0,
                crystal radius0, ionic radius0),
         (element1, 3, coordination1, ...), ...]

    Parameters
    ----------
    :symbol (string): the chemical symbol, e.g. 'Li'.
    :charge (int): the charge on the ion
    :coordination (int or string): the coordination of the ion. Can be an
                         integer or one of:
                         I, II, III, IIIPY, IV, IVSQ, V, VI, VII,
                         VIII, IX, X, XI, XII, XIV
    :spinState (string): 'high' or 'low'

    Returns
    -------
    A list of ionic radii matching the specified parameters
    """
    retval = ionic_radii.data
    if symbol is not None:
        retval = [entry for entry in retval
                  if entry[0] == symbol]
    if charge is not None:
        retval = [entry for entry in retval
                  if entry[1] == charge]
    if coordination is not None:
        # creates a map of integers 1..14 to the corresponding
        # Roman numerals
        coord = dict(
            zip(range(1, 15),
                (('I',), ('II',), ('III', 'IIIPY'), ('IV', 'IVPY', 'IVSQ'),
                 ('V',), ('VI',), ('VII',), ('VIII',), ('IX',), ('X',),
                 ('XI',), ('XII',), ('XIII',), ('XIV',))))
        # in order to use the coord map, also map each Roman numeral
        # to itself (both upper and lower case, just in case the user
        # passes i, ii, etc.)
        for coordination in coord.values():
            for rep in coordination:
                coord[rep] = (rep,)
                coord[rep.lower()] = (rep,)
        # except for 3 and 4, "entry[2] in coord[coordination]" is
        # equivalent to "entry[2] == coordination", but the former
        # allows for multiple matches, e.g. III and IIIPY.
        retval = [entry for entry in retval
                  if (entry[2] in coord[coordination])]
    if spin is not None:
        retval = [entry for entry in retval
                  if (entry[3] == spin or entry[3] is None)]
    return ([x[5] for x in retval], [x[:5]+[x[6]] for x in retval])
#end 'def ionic_radii(symbol, charge, coordination, spinState=None):'

