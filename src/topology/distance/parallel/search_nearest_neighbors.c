__kernel void search_nearest_neighbors(
        __constant const int    point_size,
        __constant const float *point_cloud,
        __constant const float *point_radii,
        __constant const int    num_nodes,
        __constant const int   *indices,
        __constant const float *dividers,
        __constant const float *grid_coord,
        __global   float       *distances) {
    /*
     * Synopsis
     * --------
     *  Calculates the nearest point in *point_cloud* to each coordinate
     *  in *grid_coord*. The arrays contains the x, y, and z data
     *  in a flattened format, *e.g.*
     *  
     *  .. code: C
     *
     *      grid_coord = (x0, y0, z0, x1, y1, z1, ..., xN, yN, zN)
     *      point_cloud  = (x0, y0, z0, x1, y1, z1, ..., xN, yN, zN)
     *      point_radii  = (r0, r1, r2, ... rN)
     *
     *                  +- root (0)
     *                  |   +- left (1)
     *                  |   |   +- left (2)
     *                  |   |   |   +- left (3)
     *                  |   |   |   |   +- left (4, leaf)
     *                  |   |   |   |   |  +- right (4, leaf)
     *                  |   |   |   |   |  |  +- right (3, leaf)
     *                  |   |   |   |   |  |  |  +- right (2)
     *                  |   |   |   |   |  |  |  |   +- left (3, leaf)
     *                  |   |   |   |   |  |  |  |   |  +- right (3, leaf)
     *                  |   |   |   |   |  |  |  |   |  |  +- right (1)
     *                  |   |   |   |   |  |  |  |   |  |  |   +- left (2)
     *                  |   |   |   |   |  |  |  |   |  |  |   |
     *                  |   |   |   |   |  |  |  |   |  |  |   |
     *                  v   v   v   v   v  v  v  v   v  v  v   v
     * indices (-1, -1, -1, -1, i, j, k, -1, l, m, -1, -1, ...)
     * dividers (perpendicular to x, y, z, x, y, z, ....)
     *
     *      Each element in indices can take one of three
     *      valid values: and atom index, [0, N); an empty
     *      leaf, N; or any other number, which is taken as
     *      a new node.
     *
     * Parameters
     * ----------
     *  :(in) point_size (int): N, the number of of points in the
     *      point_cloud
     *  :(in) point_cloud (floats): atom coordinates. (3N values)
     *  :(in) point_radii (floats): radii of the atoms (N values)
     *  :(in) num_nodes (int): the number of nodes, i.e. the number
     *      of values in the *indices* and *dividers* arrays.
     *  :(in) indices (ints): flattened tree structure (shown
     *      in the *Synopsis*
     *  :(in) dividers (floats): dividing point along each axis
     *      in turn: x, y, z, x, y, z, ...
     *  :(in) grid_coord (floats): coordinates whose nearest neighbor
     *      distance to the point cloud are to be determined. (3*num. voxels)
     *  :(out) distances (floats): distance from each value in
     *      *grid_coord* to the nearest point in *point_cloud* (num. voxels)
     */

    typedef struct _Tree {
        typedef struct _Node {
            float value;          // dividing value
            Node *parent;         // parent node (for reverse traversal)
            Node *left;           // left subtree
            Node *right;          // right subtree
            const float *coord;   // coordinate
            const float *radius;  // atom radius

            _Node() : \
                value(0.0), \
                parent(NULL), \
                left(NULL), \
                right(NULL), \
                coord(NULL) {}

            ~_Node() {
                // if a branch is deleted, so is the entire subtree
                if(left) {
                    free(left);
                }
                if(right) {
                    free(right);
                }
                left = right = coord = radius = NULL;
            }

            int axis() const {
                // returns the axis that splits this branch
                int div_plane = 0;
                for(Node *node = parent; node != NULL; node = node.parent) {
                    // The leaves do not count as another level, e.g.
                    // leaves off the root node would have, as their parent,
                    // the root node, but only one cut would have been made,
                    // and that is along the x-axis.
                    if(left != NULL && right != NULL) {
                        ++div_plane;
                    }
                }
                // there are 1 fewer dividing planes than nodes, e.g.
                // root --div1-- A --div2-- B --div3-- C
                // 4 nodes: root, A, B, C
                // 3 dividing planes: div1, div2, div3
                --div_plane;
                // run cyclically through the axes
                return div_plane%3;
            }
        } Node;
        
        // member variables
        Node root;
        float max_radius;
        
        void reconstruct(const int _num_nodes, const int *_indices, const float *_dividers,
                   const int _num_points, const float *_point_cloud, const float *_point_radii) {
            // get the range of point radii
            max_radius = *_point_radii;
            for(float *rit = _point_radii; rit < _point_radii+_num_points; ++rit) {
                if(*rit > max_radius) {
                    max_radius = *rit;
                }
            }
            // reproduce the tree structure from flattened objects
            Node *branch = &root;
            for(int i = 0; i < _num_nodes; ++i) {
                // is this a leaf?
                if(_indices[i] >= 0 || _indices[i] < _num_points) {
                    branch.coord = _point_cloud + 3*_indices[i];
                    branch.radius = _point_radii + i;
                    // the last value will be the index of the
                    // rightmost node. Once we try to back out from
                    // there, we end up at one-above-the-root node,
                    // i.e. NULL
                    while(branch != NULL ||
                          branch.parent.right != NULL) {
                        branch = branch.parent;
                    }
                    if(branch == NULL) {
                        break;
                    }
                    branch = branch.right;
                    // is the leaf empty?
                } else if(_indices[i] == _num_points) {
                    // this branch is an empty branch, e.g. two (or more)
                    // points are colinear along a principle axis.
                    // and were stored in another branch
                    // otherwise, it is a new node
                } else {
                    // each branch has a dividing value between left and right
                    branch.value = _dividers[i];
                    // fill left first ...
                    if(branch.left == NULL) {
                        branch.left = (Node*) malloc(sizeof(Node));
                        branch.left.parent = branch;
                        branch = branch.left;
                        // ... then right
                    } else {
                        branch.right = (Node*) malloc(sizeof(Node));
                        branch.right.parent = branch;
                        branch = branch.right;
                    }
                }
            }
        }
        
        void nearest_neighbor(const float *q, const Node* subtree, const float const * *ppoint,
                              float const * psearch_distance=NULL, float const * pradius_at_min=NULL) {
            /*
             * Synopsis
             * --------
             * Finds the point in *subtree* closest to *q*.
             *
             * Parameters
             * ----------
             * :(in) q (DIM-float): Point the neighbor of which we are to find.
             * :(in) subtree (Node*): Node (subtree) to search for the neighbor.
             * :(out) ppoint (float**): Nearest neighbor coordinate.
             * :(out) psearch_radius (float*): Pointer to the distance separating q and
             *      its nearest neighbor. NULL
             * :(out) pradius_at_min (float*): Pointer to the radius of the minimum point
             *
             * Returns
             * -------
             * None. Output is handled as pointer objects in the parameter list.
             */
            const int DIM = 3;
            const float INFTY = 9.999e9;
            float radius_at_min = (pradius_at_min) ? *pradius_at_min : INFTY;
            float search_distance = (psearch_distance) ? *psearch_distance : INFTY;
            float const * plocal_radius_at_min = (pradius_at_min) ? pradius_at_min : &radius_at_min;
            float const * plocal_search_distance = (psearch_distance) ? psearch_distance : &search_distance;
            // leaf case
            if(subtree->left == NULL &&
               subtree->right == NULL) {
                float dr[DIM];
                float distance = INFTY;
                float test_radius = 0.0;
                float test_norm = 0.0;
                // initialize the distance
                if(psearch_distance && pradius_at_min) {
                    // the search distance, detailed below, must look for points
                    // whose center-to-center distance is greater, but who, with
                    // their larger radius, will measure closer.
                    distance = *psearch_distance + *pradius_at_min - max_radius;
                }
                // check for an empty node
                if(subtree->coord == NULL) {
                    continue;
                }
                // distance to test point in the subtree
                test_norm = 0.0;
                for(int i = 0; i < DIM; ++i) {
                    dr[i] = subtree->coord[i] - q[i];
                    test_norm += dr[i]*dr[i];
                }
                test_norm = sqrt(test_norm);
                test_radius = subtree->radius;
                // if this point is closer, shrink the search radius
                if(distance == INFTY ||
                   test_norm-test_radius < distance-radius_at_min) {
                    *ppoint = subtree->coord;
                    distance = test_norm;
                    // radius of this closer point
                    *plocal_radius_at_min = test_radius;
                    // a point with a larger radius, though its center-to-center
                    // distance is greater, may still test closer. Be sure to look
                    // for these larger-radius points.
                    *plocal_search_distance = test_norm + (max_radius - test_radius);
                }
            // node case
            } else {
                int axis = subtree->axis();
                // point lies in the left branch
                if(*(q+axis) < subtree->value) {
                    // search the left subtree first
                    nearest_neighbor(q, subtree->left, ppoint, plocal_search_distance, plocal_radius_at_min);
                    // search the right subtree next, if it extends to that side
                    if(*(q+axis)+*plocal_search_distance >= subtree->value) {
                        nearest_neighbor(q, subtree->right, ppoint, plocal_search_distance, plocal_radius_at_min);
                    }
                // point lies in the right branch
                } else {
                    // search the right branch
                    nearest_neighbor(q, subtree->right, ppoint, plocal_search_distance, plocal_radius_at_min);
                    // search the left subtree next, if the search distance extends us into that volume
                    if(*(q+axis)-*plocal_search_distance < subtree->value) {
                        nearest_neighbor(q, subtree->left, ppoint, plocal_search_distance, plocal_radius_at_min);
                    }
                }
            }
        }
    } Tree;


    // function scope variables
    const int DIM = 3;
    size_t gid = get_global_id(0);  // which voxel in the grid are we considering?
    float dr[DIM];               // elementwise distance between points
    float rsq = 0.0;             // squared distance between points
    const float *nneighbor;      // nearest neighbor for this element
    Tree tree;                   // tree structure, constructed externally and passed piecemeal
    
    tree.reconstruct(num_nodes, indices, dividers, point_size, point_cloud, point_radii);
    tree.nearest_neighbor(grid_coord[3*gid], tree.root, &nneighbor);
    for(int i=0; i<DIM; ++i) {
        dr[i] = nneighbor[i] - grid_coord[3*gid+i];
        rsq += dr[i]*dr[i]
    }
    distance[gid] = sqrt(rsq);
}

