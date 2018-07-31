#define DIM 3

inline void i_to_ijk(int *ijk, int i,
                     const int Nx,
                     const int Ny,
                     const int Nz) {
    /*
     * Converts from a flattened index in an (Nx, Ny, Nz) grid,
     * where x varies fastest, then y, then z.
     */
    ijk[2] = i / (Nx*Ny);
    i = i % (Nx*Ny);
    ijk[1] = i / Nx;
    ijk[0] = i % Nx;
}

inline void ijk_to_scaled(float *scaled,
                          const int *ijk,
                          const int Nx,
                          const int Ny,
                          const int Nz) {
    scaled[0] = (float)(ijk[0])/Nx;
    scaled[1] = (float)(ijk[1])/Ny;
    scaled[2] = (float)(ijk[2])/Nz;
}

inline void vm_dot(float *dst,
                   const float *vec,
                   __constant const float *mat) {
    /*
     * (1x3)*(3x3) vector/matrix dot product.
     */
    dst[0] = vec[0]*mat[3*0 + 0] + \
             vec[1]*mat[3*1 + 0] + \
             vec[2]*mat[3*2 + 0];
    dst[1] = vec[0]*mat[3*0 + 1] + \
             vec[1]*mat[3*1 + 1] + \
             vec[2]*mat[3*2 + 1];
    dst[2] = vec[0]*mat[3*0 + 2] + \
             vec[1]*mat[3*1 + 2] + \
             vec[2]*mat[3*2 + 2];
}

inline float v_norm(const float *vec) {
    const float *it = vec;
    float length = 0.0;
    for(it = vec; it != vec+DIM; ++it) {
        length += *it;
    }
    return sqrt(length);
}

__kernel void voxel_atom_distance(__global float *grid,
                                  const int Nx,
                                  const int Ny,
                                  const int Nz,
                                  __constant const float *pv,
                                  const int num_atoms,
                                  __constant const float *atoms_s,
                                  __constant const float *atoms_r) {
    /*
     * Synopsis
     * --------
     * Calculates the distance between each voxel in *grid*
     * and any atom, including periodic replicas.
     *
     * Parameters
     * ----------
     * :(OUT) grid (float*): Holds the distance to the nearest atom. To
     *      save space, the index in grid must be convertable to
     *      the real-space coordinate. Therefore, *grid* is assumed
     *      to be a 3D grid with x varying most quickly, then y,
     *      then z.
     * :(IN) Nx, Ny, Nz (int): The number of bins along each dimension
     *      in the grid.
     * :(IN) pv (float*): Periodicity vector of the grid, stored in row-
     *      dominant format, *i.e.* (xx, xy, xz, yx, yy, yz, zx, zy, zz).
     * :(IN) num_atoms (int): The number of atoms over which to search.
     * :(IN) atoms_s (float*): The x-, y-, and z-
     *      coordinates of each atom **in scaled coordinates** stored
     *      as a flattened array :code:`{x0, y0, z0, ..., xN, yN, zN}`.
     * :(IN) atoms_r (float*): The radii of the atoms.
     */
    const int gid = get_global_id(0);
    int i = 0;
    int j = 0;
    int ijk[DIM];
    float distance;
    float ds[DIM];
    float dx[DIM];
    float grid_scaled[DIM];
    /* set maximum distance */
    for(i = 0; i < DIM; ++i) {
        ds[i] = 1.
    }
    vm_dot(dx, ds, pv);
    grid[gid] = v_norm(dx);
    /* axis-aligned index of this element */
    i_to_ijk(ijk, gid, Nx, Ny, Nz);
    /* scaled coordinate of this element */
    ijk_to_scaled(grid_scaled, ijk, Nx, Ny, Nz);
    for(i = 0; i < num_atoms; ++i) { /* for each atom */
        for(j = 0; j < DIM; ++j) {
            /* distance, in scaled coordiates, along each dimension */
            ds[j] = atoms_s[DIM*i + j] - grid_scaled[j];
            /* roundup, e.g. account for periodicity */
            ds[j] -= floor(ds[j] + 0.5); /* ds range = [-0.5, 0.5) */
        }
        /* scaled to real coordinates */
        vm_dot(dx, ds, pv);
        /* distance, accounting for atomic radius */
        distance = v_norm(dx) - atoms_r[i];
        /* keep track of minimum distance */
        if(distance < grid[gid]) {
            grid[gid] = distance;
        }
    } /* for each atom */
}

