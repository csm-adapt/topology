#!/usr/bin/env python
"""
Stores a structured grid.
"""

# ignore global pylint warnings:
#   C0323: Operator not followed by space
#   W0621: Redefining name from outer scope
# pylint: disable=C0323
# pylint: disable=W0621

from operator import mul

import numpy as np
import warnings


class StructuredGrid(list):  # pylint: disable=R0904
    """
    Synopsis
    --------
    Base class for managing data in a structured grid.

    API
    ---
    More information for each method is available by using
    :code:`help(StructuredGrid.[method_name])`. In the following,
    :code:`sg = StructuredGrid()`.

    :sg.copy(arg): Performs a deep copy of the StructuredGrid object *arg*
                   into *sg*. Data stored in *sg* will be lost.
    :sg.ndiv: Gets/sets the number of divisions in the structured grid.
              Data in *sg* is trimmed to fit the new size (if smaller) or
              padded with zeroes (if larger).
    :sg.pv: Gets/sets the periodicity vectors. When *pv* is set, so is
            the inverse of the periodicity matrix, *ipv*. The periodicity
            should be stored in row-dominant storage, so a conversion from
            scaled coordinates, *s*, to real, *x* would look like

            .. code:: python

                x = np.dot(s, sg.pv)

    :sg.ipv: The inverse of the periodicity matrix (row-dominant storage).
             This cannot be set explicitly, but is set along with *pv*.
             Conversion from real, *x*, to scaled, *s*, coordinates would
             look like:

             .. code:: python

                 s = np.dot(x, sg.ipv)

    :sg.ijk_to_i(aai): Converts the axis-aligned index, *e.g.* (i, j, k) in
                       3D, into the corresponding flattened index.
    :sg.i_to_ijk(i): Converts a flattened index into the corresponding
                     axis-aligned index.
    :sg.index_iter(): Used in for-loops to iterate through all axis-aligned
                      index tuples, *e.g.* all (i, j, k) values in 3D.
    :sg.ijk_to_x(aai): Returns the real-space coordinate of the center of
                       the voxel (volume element) from the axis-aligned index.
    :sg.x_to_ijk(x): Returns the axis-aligned index of the voxel that
                     contains the point at *x*. **Note**: this will not be
                     a valid index if *x* lies outside the periodicity
                     vectors. To get the corresponding voxel in a periodic
                     system, use:

                     .. code:: python

                         ijk = sg.x_to_ijk((x, y, z))
                         ijk = [i-i//N*N for i, N in zip(ijk, sg.ndiv)]
    """

    @classmethod
    def flattened_to_axis_aligned(cls, index, nbins):
        """
        Synopsis
        --------
        Transforms a flattened index to higher dimensional
        storage; e.g., the 296th element in an 12 x 12 x 12
        grid has an "axis-aligned index" of (7, 0, 2). This creates
        a grid where the first is the fastest moving index;
        the second, next; etc.; the last, slowest moving.

        Parameters
        ----------
        :index (int): flattened index of the grid
        :nbins (sequence of ints): number of bins along each direction
        :returns (tuple of ints): axis-aligned index
        """
        dim = len(nbins)
        ijk = dim*[0]
        steps = [reduce(mul, nbins[:ii], 1) for ii in range(dim)]
        for j in range(dim-1, -1, -1):
            ijk[j] = index//steps[j]
            index = index % steps[j]
        return ijk

    @classmethod
    def axis_aligned_to_flattened(cls, ijk, nbins):
        """
        Synopsis
        --------
        Transforms an axis-aligned index tuple to the
        equivalent index in the flattened grid; it
        treats the first element as the fastest moving;
        the second, next; etc. This is the counterpoise
        to *flattened_to_axis_aligned*.

        Parameters
        ----------
        :ijk (sequence of ints): axis-aligned index
        :nbins (sequence of ints): number of bins along each axis
        :returns (int): flattened index
        """
        dim = len(nbins)
        steps = [reduce(mul, nbins[:ii], 1) for ii in range(dim)]
        return sum([i*step for i, step in zip(ijk, steps)])

    def __init__(self, ndiv=None, pv=None):  # pylint: disable=C0103
        super(StructuredGrid, self).__init__()
        self._ndiv = None
        self._pv = None
        self._ipv = None
        self._steps = None
        if ndiv is not None:
            self.set_ndiv(ndiv)
        if pv is not None:
            self.set_pv(pv)

    def __getitem__(self, ijk):
        """
        Synopsis
        --------
        Accesses the data in the grid. For the StructuredGrid object
        grid::

            grid[(i, j, k, ...)] ---> element at index (i, j, k, ...)

        Alternatively::

            grid[i]

        for flat access.

        Parameters
        ----------
        :ijk (int or sequence): Index of the desired voxel, as either a
            flattened index or an axis-aligned index.

        Returns
        -------
        The object at the requested index.
        """
        try:
            if isinstance(ijk, slice):
                i = ijk
            else:
                i = self.ijk_to_i(ijk)
        except TypeError:
            i = ijk
        return(super(StructuredGrid, self).__getitem__(i))

    def __setitem__(self, ijk, value):
        """
        Synopsis
        --------
        Sets the data in the grid. For the StructuredGrid object
        grid::

            grid[(i, j, k, ...)] = v

        Alternatively::

            grid[i] = v (flat access)

        Parameters
        ----------
        :ijk (int or sequence): Index of the voxel whose value is to be set.
        :value: Object that is to be stored in the grid.

        Returns
        -------
        None
        """
        try:
            if isinstance(ijk, slice):
                i = ijk
            else:
                i = self.ijk_to_i(ijk)
        except TypeError:
            i = ijk
        super(StructuredGrid, self).__setitem__(i, value)

    def copy(self, src):
        """Stores a copy of the grid in src to self."""
        self.set_ndiv(src.get_ndiv())
        self.set_pv(src.get_pv())
        self = src[:]

    def get_ndiv(self):
        """Gets the number of divisions along each axis."""
        return tuple(self._ndiv)

    def set_ndiv(self, ndiv):
        """
        Sets the number of divisions of the grid. This may be used
        safely to reshape an existing grid; the flattened order of
        the grid will be retained.
        """
        # set the step size
        self._steps = [reduce(lambda a, b: a*b, ndiv[:i], 1) \
                       for i in range(len(ndiv))]
        # resize the list
        size = reduce(lambda a, b: a*b, ndiv, 1)
        #+ grow
        if size > len(self):
            self.extend((size-len(self))*[0])
        #+ shrink
        if size < len(self):
            del self[(size-len(self)):]
        # set the new number of divisions
        self._ndiv = ndiv[:]

    ndiv = property(get_ndiv, set_ndiv,
                    doc="StructuredGrid property that gets/sets the "
                        "number of divisions of the grid.")

    def get_pv(self):
        """Gets the periodicity vectors."""
        return self._pv

    def set_pv(self, pv):  # pylint: disable=C0103
        """
        Sets the periodicity vectors of the grid.
        """
        self._pv = np.asarray(pv)
        self._ipv = np.linalg.inv(pv)

    pv = property(get_pv, set_pv,
                  doc="Gets/sets the periodicity vectors. "
                      "This also calculates the inverses of the periodicity "
                      "matrix when the periodicity vectors are set.")

    @property
    def ipv(self):
        """
        Gets the inverse of the periodicity matrix. This is useful for
        moving from real, :code:`(x, y, z)`, to scaled, :code:`(sx, sy, sz)`,
        coordinates

        .. code:: python

            np.dot((x,y,z), self.ipv)
            (sx, sy, sz)  # position of (x, y, z) in scaled coordinates.
        """
        return self._ipv

    def ijk_to_i(self, ijk):
        """Returns the tuple-index as a flattened-index."""
        return StructuredGrid.axis_aligned_to_flattened(ijk, self._ndiv)

    def i_to_ijk(self, i):
        """Returns the flattened-index as a tuple-index."""
        return StructuredGrid.flattened_to_axis_aligned(i, self._ndiv)

    def index_iter(self):
        """
        Iterates through the N-dimensional grid indices.
        """
        for i in xrange(len(self)):
            # last index is the slowest moving
            # (for last index fastest moving, use range(ndim))
            yield self.i_to_ijk(i)

    def ijk_to_x(self, ijk):
        """
        Returns the real position of the center of the voxel
        at index i, j, k, ...
        """
        if self.ndiv is None or self.pv is None:
            raise ValueError(
                'The number of bins along each axis and the periodicity '
                'must be set before the cartesian coordinate of ' + str(ijk) +
                ' can be calculated.')
        if not hasattr(ijk, '__iter__'):
            ijk = self.i_to_ijk(ijk)
        sx = np.array(ijk, dtype=np.float)+0.5  # pylint: disable=C0103
        sx /= self.ndiv                         # pylint: disable=C0103
        return np.dot(sx, self.pv)

    def x_to_ijk(self, point):
        """
        Gives the index in which a point exists.
        """
        if self._ndiv is None or self._pv is None:
            return None
        ijk = [int(x) for x in
               np.floor(np.array(self._ndiv)*np.dot(point, self._ipv))]
        #                             ^                   ^
        #                             |                   |
        # number of voxels along each direction           |
        #                  fractional coordinates of the point
        if any([(x < 0 or x >= y) for x, y in zip(ijk, self._ndiv)]):
            warnings.warn('Point ' + str(tuple(point)) +
                          ' lies outside the grid.', UserWarning)
        return ijk
# class StructuredGrid(list):

if __name__ == '__main__':
    import sys

    def write_chgcar(ofs, grid):
        """Writes the grid as a VASP CHGCAR-formatted file."""
        print >>ofs, 'Test grid written by grid.py'
        print >>ofs, '  {:.12f}'.format(1.0)
        for row in grid.get_pv():
            for col in row:
                print >>ofs, ' {: .12f}'.format(col),
            print >>ofs, ''
        print >>ofs, '  1'
        print >>ofs, 'Direct'
        print >>ofs, '{: .12f} {: .12f} {: .12f}'.format(0, 0, 0)
        print >>ofs, ''
        print >>ofs, '{:d} {:d} {:d}'.format(*grid.get_ndiv())
        i = 0
        for v in grid:  # pylint: disable=C0103
            print >>ofs, ' {: .12f}'.format(v),
            i += 1
            if i % 5 == 0:
                print >>ofs, ''
    # end 'def write_chgcar(ofs, grid):'

    grid = StructuredGrid()  # pylint: disable=C0103

    pv = np.eye(3)           # pylint: disable=C0103
    pv[0, 0] = 2.
    pv[1, 1] = 3.
    pv[2, 2] = 4.

    grid.set_ndiv((20, 30, 40))
    grid.set_pv(pv)

    size = float(len(grid))  # pylint: disable=C0103
    for i in xrange(len(grid)):
        grid[i] = i/size

    ofs = open('index.chgcar', 'w')  # pylint: disable=C0103
    write_chgcar(ofs, grid)
    ofs.close()

    for ijk in grid.index_iter():
        point = grid.ijk_to_x(ijk)
        #print >>sys.stderr, "ijk:", ijk
        #print >>sys.stderr, "point:", point
        grid[ijk] = np.linalg.norm(point)

    ofs = open('point.chgcar', 'w')  # pylint: disable=C0103
    write_chgcar(ofs, grid)
    ofs.close()

    from random import random

    with warnings.catch_warnings() as w:
        warnings.simplefilter('always')
        for i in range(20):
            point = np.dot([1.5*random() for j in range(3)], pv)
            ijk = grid.x_to_ijk(point)
            print >>sys.stderr, "point:", point, "---> ijk:", ijk, \
                "--->", grid.ijk_to_x([x-0.5 for x in ijk]), \
                "--->", grid.ijk_to_x([x+0.5 for x in ijk])
