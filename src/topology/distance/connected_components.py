#!/usr/bin/env python
"""
Identifies the connected components in a grid. This treats all
non-zero (default) voxels, along with its non-zero neighbors (in one of
the 3**dim-1 neighboring hypercubes), as a single component.
"""

from bitarray import bitarray
from topology.distance.grid import StructuredGrid
from operator import gt, ne
import numpy as np
import sys
import time


def bind_2nd(binary_functor, value):
    """Bind *value* as the second argument in a binary_functor"""
    def unary_function(x):                      # pylint: disable=C0103
        """Unary function equivalent to binary_function(x, value)."""
        return binary_functor(x, value)         # pylint: disable=C0103
    return unary_function


class ConnectedComponents(object):
    """
    Synopsis
    --------
    Connected components are defined as regions in a :code:`StructuredGrid`
    that are spatially contiguous, *i.e.* the voxels share an face, edge,
    or corner. The :code:`ConnectedComponents` object identifies and manages
    access to tese structures.

    API
    ---
    Throughout this description, :code:`cc = ConnectedComponents()`

    :len(cc): Returns the number of connected components object that have
        been found in this grid.

    :cc[i]: Returns a list of the real-space coordinates of the voxels
        that make up connected component *i*.

    :cc.get_xyz(i): Identical to :code:`cc[i]`.

    :cc.get_ijk(i): Returns a list of the axis-aligned indices of the voxels
        thatmake up connected component *i*.

    :cc.grid: Gets/sets the :code:`StructuredGrid` object associated with
        this object. All existing connected components are invalidated.

    :cc.get_mask(i): Returns a :code:`bitarray` mask that defines which
        voxel in the grid is (1) or is not (0) part of connected component
        *i*.

    :cc.fractional_range(i): Returns the percentage of the grid spanned by
        connected component *i*.

    :cc.range(i): Returns the real-space range spanned by the bounding
        box surrounding connected component *i*, *i.e.* the diagonal
        of this bounding box.

    :cc.run(filled_test): Each voxel is determined *filled* or *unfilled*
        based on *filled_test*. A list of connected components are identified
        from the spatially contiguous range of filled voxels.
    """
    def __init__(self):
        self._grid = None
        self._connected_masks = []

    def __len__(self):
        """Returns the number of connected components."""
        return len(self._connected_masks)

    def __getitem__(self, index):
        """
        Gets the real-space positions for the connected components at
        *index*.
        """
        return self.get_xyz(index)

    def __setitem__(self, index, arg):
        """
        This does nothing, as all connected components are identified
        when *run* is called.
        """
        pass

    def __delitem__(self, index):
        """
        Removes the requested connected component from consideration.
        """
        del self._connected_masks[index]

    def get_xyz(self, index):
        """
        Returns a list of the real-space positions for the connected
        component at *index*.

        :param index: connected component index
        :type index: int or slice
        :returns: item (or list of items, if slice) at INDEX
        """
        def get_pos(mask):
            """
            Gets the voxel positions (real coordinates)
            where mask is True.
            """
            size = len(mask)
            # ijk_to_x will actually handle either a flattened index
            # or a axis-aligned tuple of indices, despite the name...
            return [self._grid.ijk_to_x(i)
                    for i, j in zip(range(size), mask) if j]

        if isinstance(index, slice):
            return [get_pos(mask) for mask in self._connected_masks[index]]
        else:
            return get_pos(self._connected_masks[index])

    def get_ijk(self, index):
        """
        Returns a list of the indices for the connected component
        at *index*.

        :param index: connected component index
        :type index: int or slice
        :returns: list of indices (or list of list of indices, if slice)
        """
        def get_pos(mask):
            """
            Gets the voxel positions (real coordinates)
            where mask is True.
            """
            size = len(mask)
            return [self._grid.i_to_ijk(i)
                    for i, j in zip(range(size), mask) if j]

        if isinstance(index, slice):
            return [get_pos(mask) for mask in self._connected_masks[index]]
        else:
            return get_pos(self._connected_masks[index])

    def get_grid(self):
        """Gets the base grid object."""
        return self._grid

    def set_grid(self, grid):
        """
        Sets the base grid object. No calculation is performed,
        however, until "run" is called.
        """
        if not isinstance(grid, StructuredGrid):
            TypeError('Connected components requires a grid object')
        self._grid = grid
        del self._connected_masks[:]

    grid = property(get_grid, set_grid,
                    doc="Gets/sets the grid associated with "
                        "this connected components object.")

    def get_mask(self, index):
        """
        Returns the connected component mask at INDEX.
        """
        return self._connected_masks[index]

    def fractional_range(self, index):
        """
        Returns the fractional range of the connected component
        at INDEX along each direction vector.

        :param index: index of the connected component of interest
        :type index: int
        :returns: fraction spanned by the component at INDEX
                  along each direction.
        :rtype: tuple of floats
        """
        mask = self._connected_masks[index]
        grid = self._grid
        filled = [bitarray(n*[False]) for n in grid.ndiv]
        for i in xrange(len(mask)):
            if mask[i]:
                ijk = grid.i_to_ijk(i)
                for j in range(len(ijk)):
                    filled[j][ijk[j]] = True
        fraction = [float(sum(f))/len(f) for f in filled]
        return tuple(fraction)

    def range(self, index):
        """
        Returns the range (distance units) of the connected
        component at INDEX along each direction.

        Parameter
        ---------
        :index (int): index of the connected component of interest

        Returns
        -------
        The distance spanned by the component at *index* along each
        direction as a tuple of floats.
        """
        frac = self.fractional_range(index)
        dist = np.dot(frac, self._grid.get_pv())
        return tuple(dist)

    def run(self, filled_test=bind_2nd(ne, 0.0)):  # pylint: disable=R0912
        """
        Generates the list of connected components in the base grid.
        """
        # ensure that the grid has been set
        if not isinstance(self._grid, StructuredGrid):
            TypeError('A grid object must set before the connected '
                      'components can be found!')
        grid = self._grid  # local reference (not copy)
        size = len(grid)

        # which bins are filled
        filled = bitarray(size)
        for i in xrange(size):
            filled[i] = filled_test(grid[i])

        # which have been checked?
        checked = bitarray(size)
        # pylint claims that bitarray objects do not have a *setall*
        # member. This is not true, at least on my system, and others
        # on which I've used this code. Revisit if this is a problem.
        try:
            checked.setall(False)  # pylint: disable=E1101
        except AttributeError:
            for i in xrange(size):
                checked[i] = False

        def check_neighbors(index):
            """
            Checks whether any periodic neighbors of the voxel at *index*
            are filled, creating a list of connected components
            """
            # dim (variable) in this function  is the dimensionality
            # of the grid, e.g. a 3D grid or a 2D grid.
            dim = len(grid.ndiv)
            # convert the 3**dim flattened-indices into a
            # dim-dimensional array. Doing this once saves
            # reproducing this calculation 3**dim times.
            ijk_neighbor = []
            for j in range(3**dim):
                ijk = dim*[0]  # (0, 0, ..., 0)
                # convert the relative indices of the neighboring
                # voxels into a dim-dimensional array with the first
                # dimension varying the fastest (-1 --> 0 --> 1), then
                # the second, etc. with the last dimension varying the most
                # slowly
                for k in range(dim-1, -1, -1):
                    ijk[k] = j // 3**k
                    j -= ijk[k] * 3**k
                    # from (0, 1, 2) to (-1, 0, 1) along each axis
                    ijk[k] -= 1
                ijk_neighbor.append(tuple(ijk))
            # this voxel has now been checked
            checked[index] = True
            try:
                connected = []
                filled_indices = [index]
                i = filled_indices.pop()
                while True:
                    # check neighboring voxels
                    ijk = grid.i_to_ijk(i)
                    connected.append(ijk)
                    for ijk_nn in ijk_neighbor:
                        # move from relative to absolute voxel tuple-indices
                        ijk_nn = [x+y for x, y in zip(ijk, ijk_nn)]
                        # handle periodicity, a value of -1 along axis i
                        # should look at N_i-1 and one at N_i should look at
                        # 0
                        ijk_nn = [x-(x//n)*n for x, n in
                                  zip(ijk_nn, grid.ndiv)]
                        ii = grid.ijk_to_i(ijk_nn)  # pylint: disable=C0103
                        # add unchecked voxels to the list of those to check
                        if not checked[ii]:
                            checked[ii] = True
                            if filled[ii]:
                                filled_indices.append(ii)
                    i = filled_indices.pop()
            except IndexError:
                return connected

        # empty any existing masks
        self._connected_masks[:] = []
        for i in xrange(size):
            if filled[i] and not checked[i]:
                connected = check_neighbors(i)
                mask = bitarray(size)
                # pylint claims that bitarray objects do not have a
                # *setall* member. This is not true, at least on my
                # system, and others on which I've used this code.
                # I have, however, wrapped this in a try block to avoid
                # problems on systems where *pylint* is correct in its
                # claim.
                try:
                    mask.setall(False)  # pylint: disable=E1101
                except AttributeError:
                    for i in xrange(size):
                        mask[i] = False
                for ijk in connected:
                    mask[grid.ijk_to_i(ijk)] = True
                self._connected_masks.append(mask)
# 'class ConnectedComponents(StructuredGrid):'


def connection_break(cc, func=gt, verbosity=0):  # pylint: disable=C0103
    """
    Synopsis
    --------
    Returns information on the value at which the connected components
    completely connect through the grid. For values on one side of the
    value returned, no connected components constructed from the
    evaluation of *func* extend through the entire length of the grid; for
    values on the other side, at least one connected component spans
    the grid along at least one dimension.

    A binary search algorithm is used to find this point, so the values
    in the connected components grid should be scalars.

    Parameters
    ----------
    :components (ConnectedComponent): Object to hold the connected components
        found that span the spatial extent of the grid. Note: *components*
        will be changed during operation, *i.e.* :code:`components.run(func)`
        will be called repeatedly, changing the connected components object.
    :func (callable): A binary function that takes the value from the grid
        on the left and the test value on the right. Values greater than the
        successful test value will lead to connected components that do
        not span the grid; values below will span the grid.
    :verbosity (int): The level of output to print to the stderr.

    Returns
    -------
    (dividing value, nsteps), where *nsteps* were required to find the
    dividing value
    """
    # pylint: disable=R0914
    # local reference to the connected component grid
    grid = cc.get_grid()
    dim = len(grid.ndiv())  # dimensionality of the grid, i.e. 2D, 3D,...
    # slightly smaller than the discretization along each direction
    epsilon = np.array([1./(2.*n) for n in grid.ndiv])
    full_range = np.ones(dim)-epsilon
    # extremes of the range of values stored in grid
    lo, hi = min(grid), max(grid)           # pylint: disable=C0103
    # mask parities, i.e. the sum of connected components in each mask
    prev_mask_sum = []
    mask_parity = False
    # continue while the distance between all lo and hi ranges is greater
    # than epsilon, i.e. stop as soon as ANY range is connected through
    nsteps = 0
    if verbosity > 0:
        start = time.clock()
    while not mask_parity:
        nsteps += 1
        mid = (hi+lo)/2.
        # report progress
        if verbosity > 0:
            end = time.clock()
            if ((verbosity > 0 and nsteps % 10 == 0) or
                (verbosity > 1 and nsteps % 1 == 0)):
                print >> sys.stderr, "connection break: (%d, %f, %.3fs)" % \
                    (nsteps, mid, (end-start)/nsteps)
        test = bind_2nd(func, mid)
        cc.run(test)
        max_frac = np.zeros(len(epsilon))
        for i in range(len(cc)):
            frac = cc.fractional_range(i)
            max_frac[:] = [max(a, b) for a, b in zip(frac, max_frac)]
        # check if any connected through
        if any(max_frac > full_range):
            lo = mid                            # pylint: disable=C0103
        # none connected through
        else:
            hi = mid                            # pylint: disable=C0103
        # check if the connected components changed with the last division
        # This check relies on the use of inequalities in the fill
        # function, so that, for example, a value that tests true for
        # as less than X will, if X increases, still test true.
        mask_sum = [sum(cc.get_mask(i)) for i in range(len(cc))]
        if prev_mask_sum:
            mask_parity = (len(prev_mask_sum) == len(mask_sum)) and \
                all([(x == y) for x, y in zip(mask_sum, prev_mask_sum)])
        prev_mask_sum[:] = mask_sum[:]
        # check if that the hi/lo spacing is not overly fine
        #if all((hi-lo) < 0.1*epsilon*edges):
            #break
    if verbosity > 0:
        end = time.clock()
        ss = end-start                          # pylint: disable=C0103
        mm = ss//60                             # pylint: disable=C0103
        ss = ss % 60                            # pylint: disable=C0103
        print >> sys.stderr, "connection break: %d:%06.3f" % (mm, ss)
    return ((hi+lo)/2, nsteps)
#end 'def connection_break(cc, func=lambda a,b: a > b, verbosity=0):'
