#include <assert.h>
// assert() is only supported for devices of compute capability 2.0 and higher
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef  assert
#define assert(arg)
#endif

/*
 * Returns the 2-norm of the vector, v[len].
 */
__device__ float v_norm(const float *v, const int len);

/*
 * Evaluates s = a*x + y, where
 *   s = 1 x xcols (row vector)
 *   a = 1 x acols (row vector)
 *   x = acols x xcols (matrix, row-dominant -- i.e. C-style)
 *   y = 1 x xcols (row vector, or NULL)
 */
__device__ void vm_saxpy(float *s, \
        const float *a, \
        const int acols, \
        const float *x, \
        const int xcols, \
        const float *y);

/*
 * Evaluates s = a*x + y, where
 *   s = arows x xcols (matrix, row-dominant -- i.e. C-style)
 *   a = arows x acols (matrix, ibid)
 *   x = acols x xcols (matrix, ibid)
 *   y = arows x xcols (matrix, ibid, or NULL)
 */
__device__ void mm_saxpy(float *s, \
        const float *a, \
        const int arows, \
        const int acols, \
        const float *x, \
        const int xcols, \
        const float *y);

/*
 * Converts the flattened index, i, into the axis-aligned
 * index, ijk, assuming axis 0 varies most quickly, then
 * axis 1, then axis 2, and so forth.
 *   ndiv = number of divisions along each dimension.
 */
__device__ void i_to_ijk(int *ijk, int i, const int *ndiv);


/*
 * Locks and mechanisms for accessing specific locks
 */
//__device__ int *grid_locks = NULL; // grid locks
__device__ int global_lock = 0; // unlocked
__device__ int acquire_lock(int* lock);
__device__ void release_lock(int* lock);

// for ensuring the grid data has been initialized before
// searching for the minimum.
__device__ int initialized = 0; // initialization steps completed?

// timing
// Always end with MAX_TIMING
enum { INIT=0, ALLOC, SYNC, INDEX, DIST, RAD, REDUCE, MAX_TIMING };
__device__ int block_timing[] = { 0, 0, 0, 0, 0, 0, 0 };

#ifndef DIM
#define DIM 3
#endif
/*
 * Calculate the distance with each voxel handled by one (or more)
 * blocks. The number of threads per block is device dependent, but
 * my current card can handle 1024 threads per block. If each thread
 * handles a single dimension (x, y, z) for an atom, this equals
 * 1024/3 = 341 atoms per block. Therefore, if there are more than
 * 341 atoms in my crystal, then I must use more than one block for
 * each voxel. What follows is the grid and block structure for this
 * code:
 *
 *   block = (max_threads_per_block, 1, 1)
 *   grid = (num_voxels, 1 + 3*num_atoms/max_threads_per_block, 1)
 */
extern __shared__ float distance_BpV_array[];
__global__ void distance_BpV(\
        float *grid, \
        const int n_grid, \
        const float *pv, \
        const int *ndiv, \
        const float *scaled_atom_pos, \
        const float *atomic_radii, \
        const int n_atoms) {
    // One (or more) blocks handle a voxel, e.g. Blocks per Voxel (BpV)
    // distance between an atom and a voxel
    const int vid = blockIdx.x;
    const int bid = blockIdx.y;
    const int tid = threadIdx.x + \
                    blockDim.x * threadIdx.y + \
                    blockDim.x * blockDim.y * threadIdx.z;
    const int BLOCK_SIZE = blockDim.x * blockDim.y * blockDim.z;
    const int MAX_ATOMS = BLOCK_SIZE / DIM;
    const int MAX_THREADS = DIM*MAX_ATOMS;
    const int ATOMS_THIS_BLOCK = max(min(MAX_ATOMS, n_atoms - MAX_ATOMS*bid), 0);
    const int THREADS_THIS_BLOCK = DIM*ATOMS_THIS_BLOCK;

    // shared array variables
    float *dx = distance_BpV_array;
    float *vec = &dx[THREADS_THIS_BLOCK];
    float *dst = &vec[DIM];

    int i = 0, j = 0;
    int ijk[DIM];
    float voxel_scaled = 0.0;
    // timing
    clock_t start = 0;
    clock_t stop = 0;

    start = clock();
    if (tid == 0 && vid < n_grid) {
        // initialize the grid
        for (i = 0; i < DIM; ++i)
            vec[i] = 1.0f;
        vm_saxpy(dst, vec, DIM, pv, DIM, NULL);
        grid[vid] = v_norm(dst, DIM);
    }
    stop = clock();
    atomicAdd(&block_timing[INIT], stop-start);
    
    start = clock();
    __syncthreads();
    stop = clock();
    atomicAdd(&block_timing[SYNC], stop-start);

    // flattened grid index, vid, to axis-aligned indices, ijk
    start = clock();
    i_to_ijk(ijk, vid, ndiv);
    stop = clock();
    atomicAdd(&block_timing[INDEX], stop-start);

    // along which direction of which atom are we working?
    start = clock();
    dx = distance_BpV_array;
    dx[tid] = scaled_atom_pos[tid + bid*MAX_THREADS];
    if (tid < THREADS_THIS_BLOCK) {
        if (tid+MAX_THREADS*bid < DIM*n_atoms) {
            // axis aligned to scaled coordinates
            j = i%DIM;
            voxel_scaled = (float)(ijk[j])/(float)(ndiv[j]);
            // which atom are we considering?
            // (scaled coordinates)
            dx[tid] = voxel_scaled - dx[tid];
            // roundup, i.e. ensure all distances lie in [-0.5, 0.5)
            // (scaled coordinates)
            dx[tid] -= floor(dx[tid]+0.5);
        }
    }
    stop = clock();
    atomicAdd(&block_timing[DIST], stop-start);

    start = clock();
    __syncthreads();
    stop = clock();
    atomicAdd(&block_timing[SYNC], stop-start);

    // once all distances have been calculated...
    start = clock();
    if (tid < THREADS_THIS_BLOCK) {
        // index of the (start of the) atom in this block
        i = tid/3;
        if (tid+MAX_THREADS*bid < DIM*n_atoms && tid%3 == 0) {
            // convert from scaled to real coordinates
            vm_saxpy(dst,
                     &dx[tid], DIM,
                     pv, DIM, NULL);
            // find the distances to each atom
            // and store in the first index of each
            // atom. At the end of this loop:
            // dx = {r0, y0, z0, r1, y1, z1, ..., rN, yN, zN}
            // note: r, not x. The x-coordinate has been lost.
            dx[tid] = v_norm(dst, DIM) - atomic_radii[i + bid*MAX_THREADS];
        }
    }
    stop = clock();
    atomicAdd(&block_timing[RAD], stop-start);

    start = clock();
    __syncthreads();
    stop = clock();
    atomicAdd(&block_timing[SYNC], stop-start);

    /* reduce algorithm to find the minimum dr */
    start = clock();
    // put all radii at the beginning
    if (tid%3 == 0) {
        i = tid/3;
        for (j = ATOMS_THIS_BLOCK/2; j > 0; j >>= 1) {
            // if there are an odd number of atoms
            // in this block, search one-past to
            // make sure to keep the odd-atom-out
            if (j > 1)
                j += (j & 0x01);
            // in case odd, do not try to access
            // one-past-the-end. Because distances
            // are stored in every DIM indices, sort
            // considers only these DIM*i indices.
            if ((i < j) && (i+j < ATOMS_THIS_BLOCK))
                dx[DIM*i] = min(dx[DIM*i], dx[DIM*(i+j)]);
            __syncthreads();
        }
    }
    stop = clock();
    atomicAdd(&block_timing[REDUCE], stop-start);

    // Store the minimum distance
    if (tid == 0 && vid < n_grid)
        grid[vid] = min(grid[vid], dx[0]);

    start = clock();
    __syncthreads();
    stop = clock();
    atomicAdd(&block_timing[SYNC], stop-start);
}

__device__ float v_norm(const float *v, const int len) {
    int i = 0;
    float rval = 0.0;
    for (i = 0; i < len; ++i) {
        rval += v[i]*v[i];
    }
    return sqrtf(rval);
}

__device__ void vm_saxpy(float * s, \
        const float * a, \
        const int acols, \
        const float * x, \
        const int xcols, \
        const float * y) {
    int i = 0;
    int j = 0;
    assert(s != a);
    for (j = 0; j < xcols; ++j) {
        s[j] = 0.0;
        for (i = 0; i < acols; ++i) {
            // this ensures that the destination, s, can be the same
            // as the left operator, a.
            s[j] += a[i]*x[i*xcols + j];
        }
        if (y != NULL)
            s[j] += y[j];
    }
}

__device__ void mm_saxpy(float *s, \
        const float *a, \
        const int arows, \
        const int acols, \
        const float *x, \
        const int xcols, \
        const float *y) {
    int i = 0;
    for (i = 0; i < arows; ++i) {
        vm_saxpy(&s[i*xcols],
                 &a[i*acols], acols, \
                 x, xcols, \
                 (y == NULL) ? NULL : &(y[i*xcols]));
    }
}

__device__ void i_to_ijk(int *ijk, int index, \
        const int *ndiv) {
    int i = 0;
    int stride[DIM];

    stride[0] = 1;
    for (i = 1; i < DIM; ++i) {
        stride[i] = stride[i-1]*ndiv[i-1];
    }
    for (i = DIM; i != 0; --i) {
        ijk[i-1] = index/stride[i-1];
        index %= stride[i-1];
    }
}

__device__ int acquire_lock(int* lock) {
    while (atomicExch(lock, 1) == 1);
    return 1;
}

__device__ void release_lock(int* lock) {
    atomicExch(lock, 0);
}

/* -------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// for easy allocation/deallocation
typedef struct __dynamic_ptrs {
    void **__host;
    void **__device;
} dynamic_ptrs;

// declarations
void* new_host(dynamic_ptrs*, size_t);
void* new_device(dynamic_ptrs*, size_t);
void cleanup(dynamic_ptrs*);
void construct(dynamic_ptrs**);
void destroy(dynamic_ptrs**);

// new object on the host
void* new_host(dynamic_ptrs *self, size_t nbytes) {
    void *ptr = malloc(nbytes);
    if (self && ptr) {
        // store this pointer in a list
        if (self->__host) {
            size_t nptrs = 0;
            char **iit = NULL;
            char **dst = NULL;
            // how many pointers
            nptrs = 0;
            for (iit = (char**) self->__host; *iit != NULL; ++iit)
                ++nptrs;
            // copy pointers
            dst = (char**) malloc((nptrs+2)*sizeof(char*));
            memcpy(dst, self->__host, nptrs*sizeof(void*));
            free(self->__host);
            // add new pointer
            dst[nptrs++] = (char*) ptr;
            // NULL terminator
            dst[nptrs++] = NULL;
            // copy back
            self->__host = (void**) malloc(nptrs*sizeof(void*));
            memcpy(self->__host, dst, nptrs*sizeof(void*));
            free(dst);
        } else {
            char **iit = NULL;
            self->__host = (void**) malloc(2*sizeof(void*));
            iit = (char**) self->__host;
            iit[0] = (char*) ptr;
            iit[1] = NULL;
        }
    }
    return ptr;
}

// new object on the device
void* new_device(dynamic_ptrs *self, size_t nbytes) {
    void *ptr = NULL;
    cudaError_t rval;
    rval = cudaMalloc(&ptr, nbytes);
    if (rval != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(rval));
        printf("          : Failed to allocate %lu bytes, e.g. %.2f floats or %.2f ints\n", \
               nbytes, \
               (float) nbytes/sizeof(float), \
               (float) nbytes/sizeof(int));
        printf("BAILING OUT!\n");
        cleanup(self);
        exit(1);
    }
    if (self && ptr) {
        // store this pointer in a list
        if (self->__device) {
            size_t nptrs = 0;
            char **iit = NULL;
            char **dst = NULL;
            // how many pointers
            nptrs = 0;
            for (iit = (char**) self->__device; *iit != NULL; ++iit)
                ++nptrs;
            // copy pointers
            dst = (char**) malloc((nptrs+2)*sizeof(char*));
            memcpy(dst, self->__device, nptrs*sizeof(void*));
            free(self->__device);
            // add new pointer
            dst[nptrs++] = (char*) ptr;
            // NULL terminator
            dst[nptrs++] = NULL;
            // copy back
            self->__device = (void**) malloc(nptrs*sizeof(void*));
            memcpy(self->__device, dst, nptrs*sizeof(void*));
            free(dst);
        } else {
            char **iit = NULL;
            self->__device = (void**) malloc(2*sizeof(void*));
            iit = (char**) self->__device;
            iit[0] = (char*) ptr;
            iit[1] = NULL;
        }
    }
    return ptr;
}

// cleanup function
void cleanup(dynamic_ptrs *self) {
    if (self->__host) {
        char **ptr = (char**) self->__host;
        for (; *ptr != NULL; ++ptr)
            free(*ptr);
        free(self->__host);
        self->__host = NULL;
    }
    if (self->__device) {
        char **ptr = (char**) self->__device;
        for (; *ptr != NULL; ++ptr)
            cudaFree(*ptr);
        free(self->__device);
        self->__device = NULL;
    }
}

// construct a new structure of dynamic pointers
void construct(dynamic_ptrs **dst) {
    *dst = (dynamic_ptrs*) malloc(sizeof(dynamic_ptrs));
    (*dst)->__host = NULL;
    (*dst)->__device = NULL;
}

// destroy a new structure of dynamic pointers
void destroy(dynamic_ptrs **dst) {
    if (dst && *dst) {
        cleanup(*dst);
        free(*dst);
        *dst = NULL;
    }
}

void check_cuda(int id, dynamic_ptrs *ptrs = NULL) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("%d. CUDA error: %s\n", id, cudaGetErrorString(error));
        if (ptrs)
            cleanup(ptrs);
        exit(1);
    }
}

void failed(const char *msg) {
    int i = 0;
    for (i = 0 ; i < strlen(msg) + 13; ++i)
        printf("-");
    printf("\n");
    printf("TEST FAILED: %s\n", msg);
    printf("\n");
}

void passed(const char *msg) {
    int i = 0;
    for (i = 0 ; i < strlen(msg) + 13; ++i)
        printf("-");
    printf("\n");
    printf("TEST PASSED: %s\n", msg);
    printf("\n");
}

void started(const char *msg) {
    int i = 0;
    for (i = 0 ; i < strlen(msg) + 14; ++i)
        printf("=");
    printf("\n");
    printf("STARTED TEST: %s\n", msg);
    for (i = 0 ; i < strlen(msg) + 14; ++i)
        printf("-");
    printf("\n");
}

__device__ float dfloat;

int test_v_norm();
int test_vm_saxpy();
int test_mm_saxpy();
int test_i_to_ijk();
int test_distance_BpV();
__global__ void call_v_norm(const float *v, const int len) {
    dfloat = v_norm(v, len);
}
__global__ void call_vm_saxpy(float *s, const float *a, const int acols, const float *x, const int xcols, const float *y) {
    vm_saxpy(s, a, acols, x, xcols, y);
}
__global__ void call_mm_saxpy(float *s, const float *a, const int arows, const int acols, const float *x, const int xcols, const float *y) {
    mm_saxpy(s, a, arows, acols, x, xcols, y);
}
__global__ void call_i_to_ijk(int *ijk, int i, const int *ndiv) {
    i_to_ijk(ijk, i, ndiv);
}

int main(int argc, const char *argv[]) {
    int i = 0;
    int nDevices = 0;
    struct cudaDeviceProp deviceProps;

    started("System Info");

    // information for device(s)
    cudaGetDeviceCount(&nDevices);
    for (i = 0; i < nDevices; ++i) {
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
    }
    cudaSetDevice(i-1); // select the last device
    check_cuda(1);

    // tests
    if (test_v_norm()) failed("v_norm");
    else passed("v_norm");

    if(test_vm_saxpy()) failed("vm_saxpy");
    else passed("vm_saxpy");

    if(test_mm_saxpy()) failed("mm_saxpy");
    else passed("mm_saxpy");

    if(test_i_to_ijk()) failed("i_to_ijk");
    else passed("i_to_ijk");

    if(test_distance_BpV()) failed("distance_BpV");
    else passed("distance_BpV");
}

int test_v_norm() {
    int i = 0;
    int nelem = 8;
    int nbytes = nelem*sizeof(float);
    int rval = 1;
    float hsol = 0.0;
    float dsol = 0.0;
    float *hvec = NULL;
    float *dvec = NULL;

    started("v_norm");

    srandom(time(NULL));

    // host vector
    hvec = (float*) malloc(nbytes);
    for (i = 0; i < nelem; ++i)
        hvec[i] = (float)(random())/RAND_MAX;

    // host solution
    hsol = 0.0;
    for (i = 0; i < nelem; ++i)
        hsol += hvec[i]*hvec[i];
    hsol = sqrt(hsol);

    // device vector
    cudaMalloc(&dvec, nbytes); check_cuda(1);
    cudaMemcpy(dvec, hvec, nbytes, cudaMemcpyHostToDevice); check_cuda(2);
    call_v_norm<<<1, 1>>>(dvec, nelem); check_cuda(3);
    cudaMemcpyFromSymbol(&dsol, dfloat, sizeof(dsol)); check_cuda(4);

    // check the results
    //printf("vector = (%f", hvec[0]);
    //for (i = 1; i < nelem; ++i)
        //printf(", %f", hvec[i]);
    //printf(")\n");
    printf("  |v(%d)|: ", nelem);
    if (dsol != hsol) {
        printf("failed, (cpu, gpu) = (%f, %f)\n", hsol, dsol);
        rval = 1;
    } else {
        printf("passed\n");
        rval = 0;
    }

    // cleanup
    free(hvec);
    cudaFree(dvec); check_cuda(5);

    return rval;
}

int test_vm_saxpy() {
    int i = 0;
    int j = 0;
    const int m = 3, n = 5; // 1xm * mxn = 1xn
    // all vectors/matrices in row dominant (C-style) format
    float *soln = NULL;
    float *hs = NULL;
    float *ha = NULL;
    float *hx = NULL;
    float *hy = NULL;
    float *ds = NULL;
    float *da = NULL;
    float *dx = NULL;
    float *dy = NULL;
    // booleans
    int check = 0;
    int rval = 0;

    started("vm_saxpy");

    // seed the random number generator
    srandom(time(NULL));

    // allocate memory
    soln = (float*) malloc(n*sizeof(float));
    hs = (float*) malloc(n*sizeof(float));
    cudaMalloc(&ds, n*sizeof(float));
    ha = (float*) malloc(m*sizeof(float));
    cudaMalloc(&da, m*sizeof(float));
    hx = (float*) malloc(m*n*sizeof(float));
    cudaMalloc(&dx, m*n*sizeof(float));
    hy = (float*) malloc(n*sizeof(float));
    cudaMalloc(&dy, n*sizeof(float));

    // generate host data
    memset(hs, 0, n*sizeof(float));
    for (i = 0; i < m; ++i)
        ha[i] = (float)(random())/RAND_MAX;
    for (i = 0; i < m; ++i)
        for (j = 0; j < n; ++j)
            hx[i*n + j] = (float)(random())/RAND_MAX;
    for (i = 0; i < n; ++i)
        hy[i] = (float)(random())/RAND_MAX;
    // host solution, y = NULL
    for (j = 0; j < n; ++j)
        for (i = 0; i < m; ++i)
            hs[j] += ha[i]*hx[i*n + j];

    // device solution, y = NULL
    cudaMemcpy(da, ha, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, m*n*sizeof(float), cudaMemcpyHostToDevice);
    call_vm_saxpy<<<1, 1>>>(ds, da, m, dx, n, NULL);
    cudaMemcpy(soln, ds, n*sizeof(float), cudaMemcpyDeviceToHost);

    // check for y = NULL
    check = 0;
    for (i = 0; i < n; ++i)
        if (hs[i] != soln[i])
            check = 1;
    printf("  s = a*x: ");
    if (check) {
        printf("failed\n");
        rval = 1;
    } else {
        printf("passed\n");
    }

    // host solution, y != NULL
    for (i = 0; i < n; ++i)
        hs[i] += hy[i];

    // device solution y != NULL
    cudaMemcpy(dy, hy, n*sizeof(float), cudaMemcpyHostToDevice);
    call_vm_saxpy<<<1, 1>>>(ds, da, m, dx, n, dy);
    cudaMemcpy(soln, ds, n*sizeof(float), cudaMemcpyDeviceToHost);

    // check for y != NULL
    check = 0;
    for (i = 0; i < n; ++i)
        if (hs[i] != soln[i])
            check = 1;
    printf("  s = a*x + y: ");
    if (check) {
        printf("failed\n");
        rval = 1;
    } else {
        printf("passed\n");
    }

    // cleanup
    free(soln);
    free(hs);
    free(ha);
    free(hx);
    free(hy);
    cudaFree(ds);
    cudaFree(da);
    cudaFree(dx);
    cudaFree(dy);

    return rval;
}

int test_mm_saxpy() {
    int i = 0;
    int j = 0;
    int k = 0;
    const int u = 3, v = 5, w = 7; // uxv * vxw = uxw
    // all vectors/matrices in row dominant (C-style) format
    float *soln = NULL;
    float *hs = NULL;
    float *ha = NULL;
    float *hx = NULL;
    float *hy = NULL;
    float *ds = NULL;
    float *da = NULL;
    float *dx = NULL;
    float *dy = NULL;
    // booleans
    int check = 0;
    int rval = 0;

    started("mm_saxpy");

    // seed the random number generator
    srandom(time(NULL));

    // allocate memory
    soln = (float*) malloc(u*w*sizeof(float));
    hs = (float*) malloc(u*w*sizeof(float));
    ha = (float*) malloc(u*v*sizeof(float));
    hx = (float*) malloc(v*w*sizeof(float));
    hy = (float*) malloc(u*w*sizeof(float));
    cudaMalloc(&ds, u*w*sizeof(float));
    cudaMalloc(&da, u*v*sizeof(float));
    cudaMalloc(&dx, v*w*sizeof(float));
    cudaMalloc(&dy, u*w*sizeof(float));

    // generate host data
    memset(hs, 0, u*w*sizeof(float));
    for (i = 0; i < u; ++i)
        for (j = 0; j < v; ++j)
            ha[i*v + j] = (float)(random())/RAND_MAX;
    for (i = 0; i < v; ++i)
        for (j = 0; j < w; ++j)
            hx[i*w + j] = (float)(random())/RAND_MAX;
    for (i = 0; i < u; ++i)
        for (j = 0; j < w; ++j)
            hy[i*w + j] = (float)(random())/RAND_MAX;
    // host solution, y = NULL
    for (i = 0; i < u; ++i)
        for (j = 0; j < v; ++j)
            for (k = 0; k < w; ++k)
                hs[i*w + k] += ha[i*v + j]*hx[j*w + k];

    // device solution, y = NULL
    cudaMemcpy(da, ha, u*v*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, v*w*sizeof(float), cudaMemcpyHostToDevice);
    call_mm_saxpy<<<1, 1>>>(ds, da, u, v, dx, w, NULL);
    cudaMemcpy(soln, ds, u*w*sizeof(float), cudaMemcpyDeviceToHost);

    // check for y = NULL
    check = 0;
    for (i = 0; i < u*w; ++i)
        if (hs[i] != soln[i])
            check = 1;
    printf("  s = a*x: ");
    if (check) {
        printf("failed\n");
        rval = 1;
    } else {
        printf("passed\n");
    }

    // host solution, y != NULL
    for (i = 0; i < u*w; ++i)
        hs[i] += hy[i];

    // device solution y != NULL
    cudaMemcpy(dy, hy, u*w*sizeof(float), cudaMemcpyHostToDevice);
    call_mm_saxpy<<<1, 1>>>(ds, da, u, v, dx, w, dy);
    cudaMemcpy(soln, ds, u*w*sizeof(float), cudaMemcpyDeviceToHost);

    // check for y != NULL
    check = 0;
    for (i = 0; i < u*w; ++i)
        if (hs[i] != soln[i])
            check = 1;
    printf("  s = a*x + y: ");
    if (check) {
        printf("failed\n");
        rval = 1;
    } else {
        printf("passed\n");
    }

    // cleanup
    free(soln);
    free(hs);
    free(ha);
    free(hx);
    free(hy);
    cudaFree(ds);
    cudaFree(da);
    cudaFree(dx);
    cudaFree(dy);

    return rval;
}

int test_i_to_ijk() {
    int i;
    const int dim = DIM;
    const int indices[] = {0, 46, 4562, 84328, -1};
    const int *it = NULL;
    int index = 0;
    int *soln = NULL;
    int *hijk = NULL;
    int *hdiv = NULL;
    int *stride = NULL;
    int *dijk = NULL;
    int *ddiv = NULL;
    int check = 0;
    int rval = 0;

    started("i_to_ijk");

    // allocate pointers
    soln = (int*) malloc(dim*sizeof(int));
    hijk = (int*) malloc(dim*sizeof(int));
    hdiv = (int*) malloc(dim*sizeof(int));
    stride = (int*) malloc(dim*sizeof(int));
    cudaMalloc(&dijk, dim*sizeof(int));
    cudaMalloc(&ddiv, dim*sizeof(int));

    // setup
    for (i = 0; i < dim; ++i)
        hdiv[i] = 45;
    stride[0] = 1;
    for (i = 1; i < dim; ++i)
        stride[i] = stride[i-1]*hdiv[i-1];

    // copy constant data
    cudaMemcpy(ddiv, hdiv, dim*sizeof(int), cudaMemcpyHostToDevice);

    // try several values
    for (it = indices; *it != -1; ++it) {
        index = *it;
        // host evaluation
        for (i = dim; i != 0; --i) {
            hijk[i-1] = index / stride[i-1];
            index %= stride[i-1];
        }

        // device evaluation
        call_i_to_ijk<<<1, 1>>>(dijk, *it, ddiv);
        cudaMemcpy(soln, dijk, dim*sizeof(int), cudaMemcpyDeviceToHost);

        // check
        check = 0;
        for (i = 0; i < dim; ++i)
            if (hijk[i] != soln[i])
                check = 1;
        printf("  %d --> (%d", *it, hijk[0]);
        for (i = 1; i < dim; ++i)
            printf(", %d", hijk[i]);
        printf("): ");
        if (check) {
            printf("failed\n");
            rval = 1;
        } else {
            printf("passed\n");
        }
    }

    // cleanup
    free(soln);
    free(hijk);
    free(hdiv);
    free(stride);
    cudaFree(dijk);
    cudaFree(ddiv);

    return rval;
}

int test_distance_BpV() {
    int i = 0;
    int j = 0;
    const int natoms = 27;
    const int Nx = 200, Ny = 200, Nz = 200;
    // dynamic pointers
    dynamic_ptrs *dyn = NULL;
    // host
    float *hgrid = NULL;
    float *hpv = NULL;
    int *hndiv = NULL;
    float *hscaled_atom_pos = NULL;
    float *hatomic_radii = NULL;
    // device
    float *dgrid = NULL;
    float *dpv = NULL;
    int *dndiv = NULL;
    float *dscaled_atom_pos = NULL;
    float *datomic_radii = NULL;
    // timing
    clock_t start, stop;

    started("distance_BpV");

    // seed random number generator
    srandom(time(NULL));

    // allocate vectors
    construct(&dyn);
    hgrid = (float*) new_host(dyn, Nx*Ny*Nz*sizeof(float));
    hpv = (float*) new_host(dyn, DIM*DIM*sizeof(float));
    hndiv = (int*) new_host(dyn, DIM*sizeof(int));
    hscaled_atom_pos = (float*) new_host(dyn, DIM*natoms*sizeof(float));
    hatomic_radii = (float*) new_host(dyn, natoms*sizeof(float));
    dgrid = (float*) new_device(dyn, Nx*Ny*Nz*sizeof(float));
    dpv = (float*) new_device(dyn, DIM*DIM*sizeof(float));
    dndiv = (int*) new_device(dyn, DIM*sizeof(int));
    dscaled_atom_pos = (float*) new_device(dyn, DIM*natoms*sizeof(float));
    datomic_radii = (float*) new_device(dyn, natoms*sizeof(float));

    // setup
    memset(hpv, 0, DIM*DIM*sizeof(float));
    for (i = 0; i < DIM; ++i)
        hpv[i*DIM + i] = 10.0;
    hndiv[0] = Nx;
    hndiv[1] = Ny;
    hndiv[2] = Nz;
    for (i = 0; i < natoms; ++i)
        for (j = 0; j < DIM; ++j)
            hscaled_atom_pos[i*DIM + j] = (float)(random())/RAND_MAX;
    for (i = 0; i < natoms; ++i)
        hatomic_radii[i] = 0.5*(float)(random())/RAND_MAX + 0.7;

    // move data to the device and execute
    cudaMemcpy(dpv, hpv, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(1, dyn);
    cudaMemcpy(dndiv, hndiv, DIM*sizeof(int), cudaMemcpyHostToDevice);
    check_cuda(2, dyn);
    cudaMemcpy(dscaled_atom_pos, hscaled_atom_pos, DIM*natoms*sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(3, dyn);
    cudaMemcpy(datomic_radii, hatomic_radii, natoms*sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(4, dyn);
    start = clock();
    distance_BpV<<<Nx*Ny*Nz, /* no./dim blocks */\
                   DIM*natoms, /* no./dim threads */\
                   (DIM*natoms+2*DIM)*sizeof(float)>>>(dgrid, Nx*Ny*Nz, \
                                                       dpv, dndiv, \
                                                       dscaled_atom_pos, datomic_radii, natoms);
    check_cuda(5, dyn);
    cudaDeviceSynchronize();
    stop = clock();
    {
        struct cudaDeviceProp devProp;
        int device_id = 0;
        float gpu_rate = 1;
        int hblock_timing[MAX_TIMING];
        float ss = (float)(stop - start)/CLOCKS_PER_SEC;
        int mm = (int)(ss/60.);
        ss -= 60.0 * mm;
        printf("  distance_BpV completed in %02d:%09.6f\n", mm, ss);
        printf("  ---------------------------------------\n");
        cudaMemcpyFromSymbol(hblock_timing, block_timing, sizeof(hblock_timing));
        cudaGetDevice(&device_id);
        cudaGetDeviceProperties(&devProp, device_id);
        gpu_rate = (float) devProp.clockRate;
        printf("     initialization: %f ms\n", hblock_timing[INIT]/gpu_rate);
        printf("     mem allocation: %f ms\n", hblock_timing[ALLOC]/gpu_rate);
        printf("    synchronization: %f ms\n", hblock_timing[SYNC]/gpu_rate);
        printf("           i_to_ijk: %f ms\n", hblock_timing[INDEX]/gpu_rate);
        printf("     cartesian dist: %f ms\n", hblock_timing[DIST]/gpu_rate);
        printf("    radial distance: %f ms\n", hblock_timing[RAD]/gpu_rate);
        printf("     minimum radius: %f ms\n", hblock_timing[REDUCE]/gpu_rate);
    }
    cudaMemcpy(hgrid, dgrid, Nx*Ny*Nz*sizeof(float), cudaMemcpyDeviceToHost);
    check_cuda(6, dyn);

    // print the data as a CHGCAR
    {
        FILE *ofs = fopen("CHGCAR", "w");
        fprintf(ofs, "%d randomly placed atoms in a %d x %d x %d grid\n", natoms, hndiv[0], hndiv[1], hndiv[2]);
        fprintf(ofs, " % 15.12f\n", 1.0);
        for (i = 0; i < DIM; ++i) {
            for (j = 0; j < DIM; ++j)
                fprintf(ofs, " % 15.12f", hpv[i*DIM + j]);
            fprintf(ofs, "\n");
        }
        fprintf(ofs, "C\n");
        fprintf(ofs, "%d\n", natoms);
        fprintf(ofs, "Selective Dynamics\n");
        for (i = 0; i < natoms; ++i) {
            for (j = 0; j < DIM; ++j) {
                fprintf(ofs, " % 12.8f", hscaled_atom_pos[i*DIM + j]);
            }
            fprintf(ofs, "\n");
        }
        fprintf(ofs, "\n");
        for (i = 0; i < DIM; ++i)
            fprintf(ofs, " %d", hndiv[i]);
        fprintf(ofs, "\n");
        for (i = 0; i < Nx*Ny*Nz; ++i)
            fprintf(ofs, "% 15.12f%s", hgrid[i], (((i+1)%5 == 0) ? "\n" : " "));
        if (i%5 != 5)
            fprintf(ofs, "\n");
        fclose(ofs);
    }

    printf("  Visually verify CHGCAR\n");

    // cleanup
    destroy(&dyn);

    return 0;
};

