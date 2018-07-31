#!/usr/bin/env python
"""
Maps the distance to the nearest neighbor to a structured grid.
"""

# Ignore specific pylint warnings
#   R0904: Too many public methods (MapBase & derived have a complex API)
# pylint: disable=R0904

import numpy as np
import os
import sys
import warnings

from ase.atoms import Atoms
from ase.data import covalent_radii, chemical_symbols, vdw_radii
from topopropy.grid import StructuredGrid
from operator import mul
from scipy.spatial import cKDTree  # pylint: disable=E0611


class MapBase(StructuredGrid):
    """
    MapBase
    =======

    :subclass of: StructuredGrid

    :Note: MapBase is a virtual class whose subclasses should overload the
           *run* function.

    Synopsis
    --------
    In addition to managing a structured grid, capability this inherits from
    *StructuredGrid*, *Map* also provides the interface to the *Atomistic
    Simulation Environment* (*ASE*) :code:`Atoms` object.

    API
    ---
    In what follows, the functionality of this class, which is intended to
    be *purely virtual*, will be detailed as if accessed from a derived
    class, *i.e.* :code:`map = MapBaseDerivedClass()`

    :MapBaseDerivedClass.covalentRadii: Dictionary of covalent radii
            accessed by chemical symbol.
    :Map.vdwRadii: Dictionary of van der Waals radii, accessed by chemical
            symbol.
    :map.atoms: Get/set the ase.atoms.Atoms object.
    :map.get_chemical_symbols(): Returns a list of the chemical symbols,
            *e.g.* ['H', 'H', 'S', 'O', 'O', 'O', 'O']. This list will be
            the same length as the number of atoms in the structure.
    :map.get_radii(): Gets the radii of the atoms.
    :map.set_radii(): These must be set
            explicitly, which can be done with relative ease by using the
            Map.covalentRadii (or analogously, Map.vdwRadii):

            .. code:: python

                map.set_radii([Map.covalentRadii[s]
                               for s in map.get_chemical_symbols()])

    :map.set_pv(pv): Sets the periodicity vectors.
    :map.run(): **virtual**
    """
    # element information
    covalentRadii = dict(zip(chemical_symbols, covalent_radii))
    vdwRadii = dict(zip(chemical_symbols, vdw_radii))

    def __init__(self, ndiv=None, pv=None):  # pylint: disable=C0103
        self._absolute = True
        self._atoms = Atoms()
        self._radii = None
        super(MapBase, self).__init__(ndiv=ndiv, pv=pv)

    @property
    def atoms(self):
        """Gets the :code:`ase.atoms` object."""
        return self._atoms

    @atoms.setter
    def atoms(self, val):
        """Sets the :code:`ase.atoms` object."""
        if not isinstance(val, Atoms):
            TypeError('The atoms must be stored as an ase Atoms object.')
        self._atoms = val
        if np.all(self.atoms.get_cell() == np.eye(3)):
            self.atoms.set_cell(self.get_pv())
        else:
            self.set_pv(self.atoms.get_cell())
        self.atoms.set_pbc(3*(True,))

    def get_chemical_symbols(self):
        """Gets the chemical symbols for the atoms stored in this grid."""
        return self._atoms.get_chemical_symbols()

    def get_radii(self):
        """
        Gets the radii for the point-of-nearest-approach calculations.
        """
        if self._radii is None:
            return len(self.atoms)*[0.0]
        else:
            return self._radii

    def set_radii(self, radii):
        """
        Sets the radii to be used for point-of-nearest-approach.
        """
        if len(radii) != len(self.get_chemical_symbols()):
            raise ValueError(
                'The number of provided radii (%d) does not '
                'match the number of atoms in the system (%d).' %
                (len(radii), len(self.get_chemical_symbols())))
        self._radii = radii[:]
        self._absolute = False

    radii = property(get_radii, set_radii,
                     doc="Sets the radii of the atoms in this map.")

    def set_pv(self, pv):  # pylint: disable=C0103
        super(MapBase, self).set_pv(pv)
        self.atoms.set_cell(pv)
        self.atoms.set_pbc(3*(True,))

    def run(self, *args, **kwds):
        """
        Because the mapping process can be expensive, it must be
        called explicitly.

        :param leafsize: Specify the leafsize for the cKDTree used in
                         finding neighboring atoms.
        :param verbosity: specify the amount of status information
                        : to write (to the stderr)
        :type leafsize: positive integer
        :type verbosity: int
        :returns: none
        """
        pass
# class MapBase(StructuredGrid):


class Map(MapBase):
    """
    :subclass of: MapBase

    Synopsis
    --------
    Creates a general-use Map object. In addition to the functionality
    detailed in the *MapBase* documentation, a fill function can, and
    must be set for Map. This fill function is used to calculate the value
    at each point in the underlying StructuredGrid.

    API
    ---
    A more detailed description of the general Map API is presented in the
    documentation for *MapBase*. As there, :code:`map = Map()`

    :map.fill_function: Gets/sets the fill function used to populate the
            StructuredGrid underlying this map. This function must take
            a single argument: the flattened index in the array. Any number
            of keywords, including zero, may be specified to provide any
            other arguments to the function.

    :map.run(**kwds): Calls the fill function for each element in the map:

        .. code:: python

            for i in xrange(len(self)):
                self[i] = self.fill_function(i, **kwds)

    """
    def __init__(self, ndiv=None, pv=None):  # pylint: disable=C0103
        self._fill_function = None
        super(Map, self).__init__(ndiv=ndiv, pv=pv)

    def get_fill_function(self):
        """Returns the fill function."""
        return self._fill_function

    def set_fill_function(self, fill_function):
        """Sets the fill function to *ff*"""
        self._fill_function = fill_function

    fill_function = property(get_fill_function, set_fill_function,
                             doc="""
        Synopsis
        --------
        Gets/sets the fill function used to populate the value of each
        voxel in the Map. This function must take one argument, an index
        corresponding to the flattened index of a voxel in the StructuredGrid
        underlying this Map. All other function parameters must be
        specified as keywords. These keywords can be set at runtime as
        keyword arguments to :code:`self.run(kwds)`.

        Parameters
        ----------
        :getter: None
        :setter: fill function, as described in the synopsis. This function
            must return only the value to be stored.
        """)

    def run(self, *args, **kwds):
        """
        Synopsis
        --------
        Populates the map based on the fill function, which must be
        defined before :code:`run` is called.

        Parameters
        ----------
        :args: Unnamed arguments, which will be ignored.
        :kwds: Named arguments to the function, which will, along with the
            index of each voxel in turn, be passed to
            :code:`self.fill_function`.

        Returns
        -------
        None
        """
        if self.fill_function is None:
            raise ValueError("The fill function must be set before the "
                             "map can be populated.")
        for i in xrange(len(self)):
            self[i] = self.fill_function(i, **kwds)
#end 'class Map(MapBase):


class DistanceMap(MapBase):
    """
    :subclass of: MapBase

    --------
    Synopsis
    --------
    Maps to a regular grid the distance between each voxel and its
    nearest neighbor, accounting for the finite radius of the atom.
    A negative value indicates the voxel lies inside the radius of
    the nearest atom.

    ---
    API
    ---
    A more detailed description of the general Map API is presented in
    documentation for MapBase.

    .. code:: python

        dmap = DistanceMap()

    :dmap.run(): Calculates the distance from each voxel in the structured
        grid to the nearest atom, accounting for the radius of each atom.
    """
    def __init__(self, ndiv=None, pv=None):  # pylint: disable=C0103
        super(DistanceMap, self).__init__(ndiv=ndiv, pv=pv)

    def run(self, *args, **kwds):
        """
        Synopsis
        --------
        Populates the grid based on the distance from each voxel
        to its nearest neighbor atom, taking into account the
        radius of each atom. This can be a (very) expensive operation,
        and so, must be called explicitly. It is parallelized using
        the :code:`multiprocessor` python module at present.

        Parameters
        ----------
        None

        Returns
        -------
        None, but the contents of the StructuredGrid underlying this Map
        will be modified.
        """

        def has_pycuda():
            """Is pycuda installed?"""
            try:
                import pycuda.driver as cuda
                return True
            except ImportError:
                return False

        def has_pyopencl():
            """Is pyopencl installed?"""
            try:
                import pyopencl as cl
                return True
            except ImportError:
                return False

        def has_multiprocessing():
            """Is multiprocessing installed?"""
            try:
                import multiprocessing as mp
                return True
            except ImportError:
                return False

        if has_pycuda():
            print ">>> Using CUDA"
            self.pycuda_parallel_run()
        elif has_pyopencl():
            print ">>> Using OpenCL"
            import pyopencl as cl  # is this necessary given "has_pyopencl"?
            devices = cl.get_platforms()[0].get_devices()
            vendors = dict([(d.vendor, i) for i, d in enumerate(devices)])
            if vendors.has_key('NVIDIA'):
                self.pyopencl_parallel_run(vendors['NVIDIA'])
            else:
                self.pyopencl_parallel_run(0)
        elif has_multiprocessing():
            print ">>> Using multiprocessing"
            self.mp_parallel_run()
        else:
            raise RuntimeError("No serial distance mapping algorithm "
                               "is available. Please install either "
                               "pycuda, pyopencl, or multiprocessing "
                               "to continue.")

    def pycuda_parallel_run(self, device_index=0):
        """
        Synopsis
        --------
        Calculates the distance between each voxel in the grid and the
        nearest neighboring atom, accounting for the radius of the atom.
        
        Parameters
        ----------
        :device_index (int): The index of the device in the list
            returned by :code:`cl.get_platforms()[i].get_devices()`

        Returns
        -------
        None, but the contents of the StructuredGrid underlying this map
        will be modified.
        """
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        print "pycuda.autoinit was called, and a context initialized"

        # get details about this device
        dev = cuda.Device(device_index)
        max_threads_per_block = dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)

        # make a numpy copy of the flattened atom positions
        atom_pos = np.array(self.atoms.get_scaled_positions(),
                              dtype=np.float32,
                              copy=True)
        atom_pos = np.reshape(atom_pos, (atom_pos.size,))
        # make a numpy copy of the atom radii
        atom_radii = np.array(self.radii, dtype=np.float32, copy=True)
        # sizes
        natoms = np.int32(atom_radii.size)
        # make a local numpy destination for the map
        grid = np.ndarray((len(self),), dtype=np.float32)
        # dimensions of the grid
        ndiv = np.array(self.get_ndiv(), dtype=np.int32, copy=True)
        # periodicity vectors as flattened index
        pv = np.array(self.pv, dtype=np.float32, copy=True).reshape((self.pv.size,))

        print "Number of atoms:", natoms

        n_threads_per_block = min(3*natoms, max_threads_per_block)
        cGrid = (int(len(self)),
                 int(1 + int(3*natoms/max_threads_per_block)),
                 1)
        cBlock = (int(n_threads_per_block), 1, 1)
        #cSMem = (3+1)*self.atoms.get_number_of_atoms()*\
        cSMem = 10*natoms*np.dtype(np.float32).itemsize
        print "CUDA grid:", cGrid
        print "CUDA block:", cBlock
        # read in the kernel
        path = os.path.dirname(__file__)
        with open(path + '/parallel/distance_kernel.cu') as ifs:
            kernel = ifs.read()
        # create device buffers
        atom_buf = cuda.mem_alloc(atom_pos.nbytes)
        radii_buf = cuda.mem_alloc(atom_radii.nbytes)
        pv_buf = cuda.mem_alloc(pv.nbytes)
        ndiv_buf = cuda.mem_alloc(ndiv.nbytes)
        # to use, at most, 90% of the free memory available
        # break up the grid into chunks that will fit on the
        # device.
        free, total = cuda.mem_get_info()
        print "memory (free, total):", free, total
        print "sizeof grid:", grid.nbytes
        grid_buf_size = min(grid.nbytes, 9*free/10)
        chunk_size = np.int32(\
                grid.size*float(grid_buf_size)/float(grid.nbytes))
        grid_chunk = np.ndarray((chunk_size,), dtype=np.float32)
        print "chunk_size:", chunk_size
        grid_buf_size = grid_chunk.nbytes
        print "grid buffer:", grid_buf_size
        grid_buf = cuda.mem_alloc(grid.nbytes)
        ngrid = np.int32(grid.size)
        #grid_buf = cuda.mem_alloc(grid_buf_size)
        # copy const data to device
        cuda.memcpy_htod(atom_buf, atom_pos)
        cuda.memcpy_htod(radii_buf, atom_radii)
        cuda.memcpy_htod(pv_buf, pv)
        cuda.memcpy_htod(ndiv_buf, ndiv)
        print "copied constant data to device"
        #print "--"
        #for i,line in enumerate(kernel.split('\n')):
            #print "%d. %s" % (i+1, line)
        #print "--"
        # create the program
        kernel = SourceModule(kernel)
        print "built kernel"
        # enqueue the kernel
        func = kernel.get_function("distance_BpV")
        cuda.memcpy_htod(grid_buf, grid)
        for ilo in range(0, grid.size, chunk_size):
            ihi = min(ilo+chunk_size, grid.size)
            print "Chunk [%d, %d)..." % (ilo, ihi),
            sys.stdout.flush()
            # move over this chunk
            #cuda.memcpy_htod(grid_buf, grid[ilo:ihi])
            print "copied"
            sys.stdout.flush()
            # run the task
            #ngrid = np.int32(ihi-ilo)
            print "N_grid: %d" % ngrid
            sys.stdout.flush()
            func(grid_buf, ngrid,
                 pv_buf, ndiv_buf,
                 atom_buf, radii_buf, natoms,
                 #grid=cGrid, block=cBlock)
                 grid=cGrid, block=cBlock, shared=cSMem)
            # copy the results from the device back to the host
            #cuda.memcpy_dtoh(grid[ilo:ihi], grid_buf)
        # copy the results from the device back to the host
        cuda.memcpy_dtoh(grid, grid_buf)
        # and now from the local copy, to this object
        self[:] = grid[:]
    
    def pyopencl_parallel_run(self, device_index):
        """
        Synopsis
        --------
        Calculates the distance between each voxel in the grid and the
        nearest neighboring atom, accounting for the radius of the atom.
        
        Parameters
        ----------
        :device_index (int): The index of the device in the list
            returned by :code:`cl.get_platforms()[i].get_devices()`

        Returns
        -------
        None, but the contents of the StructuredGrid underlying this map
        will be modified.
        """
        import pyopencl as cl

        # make a numpy copy of the flattened atom positions
        atom_pos = np.array(self.atoms.get_scaled_positions(),
                              dtype=np.float32,
                              copy=True)
        atom_pos = np.reshape(atom_pos, (atom_pos.size,))
        # make a numpy copy of the atom radii
        atom_radii = np.array(self.radii, dtype=np.float32, copy=True)
        # sizes
        natoms = np.int32(atom_radii.size)
        # make a local numpy destination for the map
        grid = np.zeros(len(self), dtype=np.float32)
        # dimensions of the grid
        Nx, Ny, Nz = [np.int32(x) for x in self.get_ndiv()]
        # periodicity vectors as flattened index
        pv = np.array(self.pv, dtype=np.float32, copy=True).reshape((self.pv.size,))
        # read in the kernel
        path = os.path.dirname(__file__)
        with open(path + '/parallel/distance_kernel.cl') as ifs:
            kernel = ifs.read()
        # set up the pyopencl environment
        ctx = cl.create_some_context([device_index])
        #ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags
        # create device buffers
        grid_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                             hostbuf=grid)
        atom_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=atom_pos)
        radii_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=atom_radii)
        pv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                           hostbuf=pv)
        # create the program
        prg = cl.Program(ctx, kernel).build()
        # enqueue the kernel and run the task
        kernel = prg.voxel_atom_distance
        kernel.set_args(grid_buf, Nx, Ny, Nz, pv_buf,
                        natoms, atom_buf, radii_buf)
        cl.enqueue_nd_range_kernel(queue, kernel, grid.shape, None)
        #prg.voxel_atom_distance(queue, len(self), None,
                #grid_buf, np.int32(Nx), np.int32(Ny), np.int32(Nz),
                #pv_buf, np.int32(natoms), atom_buf, radii_buf)
        # copy the results from the device back to the host
        cl.enqueue_copy(queue, grid, grid_buf)
        # and now from the local copy, to this object
        self[:] = grid[:]


    # A lot of housekeeping is necessary to make this method efficient.
    # Since a fill function (a.k.a. cost function) will be called for
    # each voxel in the grid, any fat in each function call will be
    # augmented hundreds-of-thousands to millions of times. The local
    # variables serve both as storage and intrinsic documentation of the
    # progression of the method. Issues of too many variables (R0914)
    # cannot be helped. The issue of excess branches (R0912) and
    # excess statements (R0915) are also unavoidable, but the basis for
    # complaint, i.e. they make the code difficult to follow, is alleviated
    # by extensive commenting.
    # pylint: disable=R0912
    # pylint: disable=R0914
    # pylint: disable=R0915
    def mp_parallel_run(self):
        """
        Synopsis
        --------
        This calculates the distance between each voxel in the
        StructuredGrid, which underlies this Map, and its nearest
        neighbor atom, accounting for the radius of the each atom.

        The code is accelerated using the :code:`multiprocessing`
        module.

        Parameters
        ----------
        None

        Returns
        -------
        None, but this method modifies the contents of the grid.
        """
        import multiprocessing as mp
        # pylint: disable=R0913
        def min_distance(out_q, index, points, pt_radii, tree, rcut):
            """
            Synopsis
            --------
            Returns the distance between the point and the nearest
            atom in a KDTree.

            Parameters
            ----------
            :out_q: multiprocessor queue to store the results
            :index (int): integer identifier for this process
            :points (matrix-like): coordinates from which the minimum
                    distance is measured
            :pt_radii (sequence of floats): the radii of each atom in the
                    the tree
            :tree (cKDTree): the KDTree construction for the atoms in the map
            :rcut (float): maximum cutoff radius for tree search

            Returns
            -------
            None, all results are pushed onto *out_q* as
            (index, [r0, r1, ..., r_npoints]).
            """
            max_radius = max(pt_radii)
            min_radius = min(pt_radii)
            data_size = len(tree.data)
            epsilon = max_radius/min_radius - 1.
            results = []
            for point in points:
                # pylint: disable=C0103
                di, ii = tree.query(point, eps=epsilon)
                # 0 or 1 neighbors
                if not hasattr(di, '__iter__'):
                    # no neighbor found
                    if ii == data_size:
                        results.append(max_radius)
                    # only one neighbor found
                    else:
                        results.append(di-pt_radii[ii])
                # several neighbors found
                else:
                    try:
                        min_value = min([x-pt_radii[i] for x, i in zip(di, ii)
                                        if i < data_size])
                        results.append(min_value)
                        #results.append(min([x-pt_radii[i] \
                                #for x,i in zip(di,ii) if i < dataSize]))
                        #results.append(np.random.random())
                        #print point, '-->', minValue
                    except ValueError:
                        warnings.warn('No neighbors were found '
                                      'within %g' % (rcut,), RuntimeWarning)
                        results.append(rcut)
            out_q.put((index, results))

        # local copies of mapping variables
        nprocs = mp.cpu_count()
        nbins = self.get_ndiv()[:]
        points = self.atoms.positions
        pt_radii = self.get_radii()[:]
        pv = self.get_pv()  # pylint: disable=C0103
        # convenience variables
        dim = len(nbins)
        npoints = len(points)
        grid_size = reduce(mul, nbins)
        # if periodic, duplicate the points into 3^dim periodic cells
        if pv is not None:
            # roundup
            ipv = np.linalg.inv(pv)
            for pt in points:           # pylint: disable=C0103
                sx = np.dot(pt, ipv)    # pylint: disable=C0103
                sx -= np.floor(sx)      # pylint: disable=C0103
                pt[:] = np.dot(sx, pv)
            # how many periodic boxes (including center cell)
            n_periodic_boxes = 3**dim
            # big array of points to hold all 3**dim copies of points
            big_points = np.zeros((n_periodic_boxes*npoints, dim))
            # also expand pt_radii, if given
            if pt_radii is not None:
                pt_radii = n_periodic_boxes*list(pt_radii)
            else:
                pt_radii = n_periodic_boxes*npoints*[0.0]
            # fill big box
            for box in xrange(n_periodic_boxes):
                # periodic box [-1, -1, -1] --> [1, 1, 1]
                ijk = [i-1 for i in
                       StructuredGrid.flattened_to_axis_aligned(box, dim*[3])]
                delta = np.dot(ijk, pv)
                for i in xrange(npoints):
                    big_points[i+box*npoints, :] = points[i]+delta
        # otherwise, store big_points <-- points so later code is clean
        else:
            pv = np.diag(np.max(points, axis=0) -  # pylint: disable=C0103
                         np.min(points, axis=0))
            big_points = np.array(points, copy=True)
            if pt_radii is None:
                pt_radii = npoints*[0.0]
        # create a KDTree for the big_points
        leafsize = max(min(2, len(big_points)/2), 1)  # system size-dependent
        tree = cKDTree(big_points, leafsize=leafsize)
        # use the largest point separation as the maximum cutoff radius
        # for neighbor searches
        max_radius = 0.0
        for pt in big_points:               # pylint: disable=C0103
            # pylint: disable=C0103
            # pylint: disable=W0612
            di, ii = tree.query(pt, k=2)
            if di[1] > max_radius:
                max_radius = di[1]
        # trim the bigPoints and pt_radii to include only points
        # beyond, at most, max_radius of the periodic boundary (in
        # addition to the central cell, of course)
        ipv = np.linalg.inv(pv)
        border_width = max(np.dot(3*[max_radius], ipv))
        mask = [all([(-border_width <= x <= 1.+border_width)
                for x in np.dot(a, ipv)])
                for a in big_points]
        pt_radii = np.array([p for p, m in zip(pt_radii, mask) if m])
        big_points = np.array([x for x, m in zip(big_points, mask) if m])
        # recreate a KDTree for the big_points
        tree = cKDTree(big_points, leafsize=leafsize)
        # calculate distances from each voxel to its nearest atom
        try:
            first = 0
            last = grid_size
            chunksize = (last-first)/nprocs + 1
            out_q = mp.Queue()
            procs = []
            for i in range(nprocs):
                ilo = first + i*chunksize
                ihi = min(first + (i+1)*chunksize, last)
                points = [self.ijk_to_x(index)
                          for index in xrange(ilo, ihi)]
                p = mp.Process(             # pylint: disable=C0103
                    target=min_distance,
                    args=(out_q, i, points, pt_radii, tree, max_radius))
                procs.append(p)
                p.start()
            # collect the results
            grid = []
            for i in range(nprocs):
                grid.append(out_q.get())
            # as they may have been processed/pushed to the queue in an
            # arbitrary order, ensure the order is finally recovered
            grid.sort(key=lambda g: g[0])

            # wait for all worker processes to finish
            for p in procs:             # pylint: disable=C0103
                p.join()
        except:
            print >> sys.stderr, 'Cleaning child processes...',
            for p in procs:             # pylint: disable=C0103
                p.terminate()
            for p in procs:             # pylint: disable=C0103
                p.join()
            print >> sys.stderr, 'done'
            raise

        # done
        self[:] = sum([g[1] for g in grid], [])
#end 'class DistanceMap(Map):'
