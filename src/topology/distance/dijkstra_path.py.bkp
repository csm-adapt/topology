# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:44:00 2013

@author: bkappes
"""

from bitarray import bitarray
from itertools import permutations
from operator import lt


# Why not use bisect and insort from the bisect module? Because I need
# to be able to specify the comparison function. By default, this orders
# in ascending order, just like bisect.bisect and bisect.insort
def bisect(a, x, comparable=lt):                    # pylint: disable=C0103
    """
    --------
    Synopsis
    --------
    Finds the index where x would be inserted into a. If x is
    already present in a, then index will be inserted to the
    right, i.e. higher index.

    ----------
    Parameters
    ----------
    :a (list): List sorted by cmp.
    :x (object): Object whose position is to be identified.
    :cmp (callable): Comparison that is to be used to identify the
                     location of the x in a.
    :returns (int): Index where x is to be inserted
    """
    # pylint: disable=C0103
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if comparable(x, a[mid]):
            hi = mid                            #
        else:
            lo = mid+1
    return lo


def insort(a, x, comparable=lt):                     # pylint: disable=C0103
    """
    --------
    Synopsis
    --------
    Insert item x in list a, and keep it sorted according
    to cmp, assuming a is already so sorted.

    If x is already in a, insert it to the right of the
    rightmost x.

    ----------
    Parameters
    ----------
    :a (list): List to be sorted
    :x (object): object that is to be inserted into a
    :cmp (callable): comparable to apply when sorting the list
    :returns: None
    """
    # pylint: disable=C0103
    i = bisect(a, x, comparable)
    a.insert(i, x)


# pylint: disable=R0912
# pylint: disable=R0914
def dijkstra_path(in_grid,
                  from_node_index,
                  to_node_index,
                  cost_func=None,
                  periodic=False):
    """
    --------
    Synopsis
    --------
    Calculates the shortest path, as defined by the cumulative
    cost function *cost_func*, going *from_node_index* *to_node_index*.

    ----------
    Parameters
    ----------
    :in_grid: StructuredGrid-like object through which the path travels.
    :from_node_index (integer): Index of the node at which the path starts.
    :to_node_index (integer): Index of the node at which the path ends.
    :cost_func (callable): Function of two variables that defines the cost
                          of moving from a node on the path (first arg)
                          to a prospective (second arg) node.
    :periodic (bool): Should a periodic path be found?
    :returns (tuple): Cost to traverse the path and the indices that traverse
              the path from *from_node_index* to *to_node_index*, including the
              end points in the path, *i.e.* (cost, indices)
    """
    # Construct the vectors that get us from a center node
    # to all adjacent nodes
    # For 3D: ((1,0,0), (0,1,0), (0,0,1))
    positive_steps = len(in_grid.ndiv)*[0]  # (0, 0, ..., 0)
    positive_steps[0] = 1  # (1, 0, ..., 0)
    positive_steps = list(set(permutations(positive_steps)))
    # For 3D: ((-1,0,0), (0,-1,0), (0,0,-1))
    negative_steps = len(in_grid.ndiv)*[0]
    negative_steps[0] = -1  # (-1, 0, ..., 0)
    negative_steps = list(set(permutations(negative_steps)))
    # both forward and backward steps
    relative_steps = negative_steps + positive_steps

    def get_adjacent_node_indices(center_node_index):
        """
        --------
        Synopsis
        --------
        Returns a tuple of indices for adjacent nodes that have yet to be
        considered central nodes.

        ----------
        Parameters
        ----------
        :center_node_index (integer): Index of the central node whose adjacent
                   nodes are sought.
        :returns (tuple of ints): Indices of adjacent nodes that have not
                   yet been central nodes.
        """
        # get axis-aligned indices for the center node
        ijk_center_node = in_grid.i_to_ijk(center_node_index)
        adjacent_node_indices = []
        # relative_steps is defined in the parent namespace to this
        # function
        for dijk in relative_steps:
            # axis-aligned indices for the prospective adjacent node
            ijk_adjacent_node = [a+b for a, b in
                                 zip(ijk_center_node, dijk)]
            if periodic:
                ijk_adjacent_node = [(i - n*(i // n)) for i, n in
                        zip(ijk_adjacent_node, in_grid.ndiv)]
            # search for adjacent nodes is not periodic, t.f. exclude
            # potential nodes that lie beyond the boundaries of the grid
            if all([0 <= i < n for i, n in
                    zip(ijk_adjacent_node, in_grid.ndiv)]):
                # axis-aligned to flattened index
                adjacent_node_index = in_grid.ijk_to_i(ijk_adjacent_node)
                # Exclude node if it has already been a center.
                # path_set, as with relative_steps, must be defined in the
                # parent namespace.
                if not path_set[adjacent_node_index]:
                    adjacent_node_indices.append(adjacent_node_index)
        return tuple(adjacent_node_indices)
    # end 'def get_adjacent_node_indices(center_node_index):'

    if not cost_func:
        cost_func = lambda i, j: in_grid[j]
    # shortest path to each node, None if node not yet in path
    path_value = len(in_grid)*[None]
    # index to the previous node in the shortest path to each node
    prev_node = len(in_grid)*[None]
    # bitarray holds those nodes that have (1) or have not (0) been
    # considered as central nodes, i.e. nodes whose path has been
    # established.
    path_set = bitarray(len(in_grid))
    try:
        path_set.setall(0)                          # pylint: disable=E1101
    except AttributeError:
        for i in xrange(len(path_set)):
            path_set[i] = False
    # queue of node indices to serve as the central node, these
    # are the lowest adjacent nodes from previous iterations
    adjacent_queue = [from_node_index]
    # keep vector sorted in descending order relative to the value
    # stored in path_value, so that the last has the shortest path

    def comparable(i, j):
        """Comparable to use when comparing path values."""
        return path_value[i] > path_value[j]

    # ----------- FORWARD ----------- #
    central = adjacent_queue.pop()
    prev_node[central] = None
    path_value[central] = 0.0
    path_set[central] = 1
    while central != to_node_index:
        for adjacent in get_adjacent_node_indices(central):
            # calc function to get to adjacent from central
            path = cost_func(central, adjacent)
            if path < 0:
                raise ValueError("A negative value for the cost function "
                                 "was encountered, but it must be "
                                 "everywhere positive. May I suggest "
                                 "f(i,j) --> exp(f(i,j)), or similar?")
            path += path_value[central]
            # has a path to adjacent already been found?
            if path_value[adjacent] is not None:
                # is the new path lower cost?
                if path < path_value[adjacent]:
                    # remove adjacent node from previous location
                    # in adjacent_queue. Why not check for 0? Because
                    # if we get here, adjacent will already be in
                    # adjacent_queue, and bisect, which returns the
                    # right insertion point, will return a value in
                    # the range [1,len(adjacent_queue)]
                    i = bisect(adjacent_queue,
                               adjacent,
                               comparable=comparable)-1
                    adjacent = adjacent_queue.pop(i)
                # no, then move to the next adjacent node
                else:
                    continue
            # (re)add adjacent to the adjacent_queue
            prev_node[adjacent] = central
            path_value[adjacent] = path
            insort(adjacent_queue,
                   adjacent,
                   comparable=comparable)
        # get the next central node, i.e. the adjacent node with the
        # cheapest path, as there is no cheaper way to get to this
        # soon-to-be-central node.
        central = adjacent_queue.pop()
        path_set[central] = 1
    # ----------- BACKWARD ----------- #
    pathway = []
    while central is not None:
        pathway += [central]
        central = prev_node[central]
    pathway.reverse()
    return (path_value[to_node_index], pathway)
#end 'def dijkstra_path(in_grid, from_node_index, to_node_index, \'
