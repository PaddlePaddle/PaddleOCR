# Author: Thomas Moreau <thomas.moreau.2010@gmail.com>
# Author: Olivier Grisel <olivier.grisel@ensta.fr>

# See quad_tree.pyx for details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# This is effectively an ifdef statement in Cython
# It allows us to write printf debugging lines
# and remove them at compile time
cdef enum:
    DEBUGFLAG = 0

cdef float EPSILON = 1e-6

# XXX: Careful to not change the order of the arguments. It is important to
# have is_leaf and max_width consecutive as it permits to avoid padding by
# the compiler and keep the size coherent for both C and numpy data structures.
cdef struct Cell:
    # Base storage structure for cells in a QuadTree object

    # Tree structure
    SIZE_t parent              # Parent cell of this cell
    SIZE_t[8] children         # Array pointing to children of this cell

    # Cell description
    SIZE_t cell_id             # Id of the cell in the cells array in the Tree
    SIZE_t point_index         # Index of the point at this cell (only defined
                               # in non empty leaf)
    bint is_leaf               # Does this cell have children?
    DTYPE_t squared_max_width  # Squared value of the maximum width w
    SIZE_t depth               # Depth of the cell in the tree
    SIZE_t cumulative_size     # Number of points included in the subtree with
                               # this cell as a root.

    # Internal constants
    DTYPE_t[3] center          # Store the center for quick split of cells
    DTYPE_t[3] barycenter      # Keep track of the center of mass of the cell

    # Cell boundaries
    DTYPE_t[3] min_bounds      # Inferior boundaries of this cell (inclusive)
    DTYPE_t[3] max_bounds      # Superior boundaries of this cell (exclusive)


cdef class _QuadTree:
    # The QuadTree object is a quad tree structure constructed by inserting
    # recursively points in the tree and splitting cells in 4 so that each
    # leaf cell contains at most one point.
    # This structure also handle 3D data, inserted in trees with 8 children
    # for each node.

    # Parameters of the tree
    cdef public int n_dimensions         # Number of dimensions in X
    cdef public int verbose              # Verbosity of the output
    cdef SIZE_t n_cells_per_cell         # Number of children per node. (2 ** n_dimension)

    # Tree inner structure
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t cell_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef public SIZE_t n_points          # Total number of points
    cdef Cell* cells                     # Array of nodes

    # Point insertion methods
    cdef int insert_point(self, DTYPE_t[3] point, SIZE_t point_index,
                          SIZE_t cell_id=*) nogil except -1
    cdef SIZE_t _insert_point_in_new_child(self, DTYPE_t[3] point, Cell* cell,
                                           SIZE_t point_index, SIZE_t size=*
                                           ) nogil
    cdef SIZE_t _select_child(self, DTYPE_t[3] point, Cell* cell) nogil
    cdef bint _is_duplicate(self, DTYPE_t[3] point1, DTYPE_t[3] point2) nogil

    # Create a summary of the Tree compare to a query point
    cdef long summarize(self, DTYPE_t[3] point, DTYPE_t* results,
                        float squared_theta=*, SIZE_t cell_id=*, long idx=*
                        ) nogil

    # Internal cell initialization methods
    cdef void _init_cell(self, Cell* cell, SIZE_t parent, SIZE_t depth) nogil
    cdef void _init_root(self, DTYPE_t[3] min_bounds, DTYPE_t[3] max_bounds
                         ) nogil

    # Private methods
    cdef int _check_point_in_cell(self, DTYPE_t[3] point, Cell* cell
                                  ) nogil except -1

    # Private array manipulation to manage the ``cells`` array
    cdef int _resize(self, SIZE_t capacity) nogil except -1
    cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1
    cdef int _get_cell(self, DTYPE_t[3] point, SIZE_t cell_id=*) nogil except -1
    cdef np.ndarray _get_cell_ndarray(self)
