import numpy as np
import copy


class Complex:
    def __init__(self, dim, func, func_args=(), symmetry=False, bounds=None,
                 g_cons=None, g_args=()):
        self.dim = dim
        self.bounds = bounds
        self.symmetry = symmetry  # TODO: Define the functions to be used
        #      here in init to avoid if checks
        self.gen = 0
        self.perm_cycle = 0

        # Every cell is stored in a list of its generation,
        # e.g., the initial cell is stored in self.H[0]
        # 1st get new cells are stored in self.H[1] etc.
        # When a cell is subgenerated it is removed from this list

        self.H = []  # Storage structure of cells
        # Cache of all vertices
        self.V = VertexCache(func, func_args, bounds, g_cons, g_args)

        # Generate n-cube here:
        self.n_cube(dim, symmetry=symmetry)

        # TODO: Assign functions to a the complex instead
        if symmetry:
            self.generation_cycle = 1
            # self.centroid = self.C0()[-1].x
            # self.C0.centroid = self.centroid
        else:
            self.add_centroid()

        self.H.append([])
        self.H[0].append(self.C0)
        self.hgr = self.C0.homology_group_rank()
        self.hgrd = 0  # Complex group rank differential
        # self.hgr = self.C0.hg_n

        # Build initial graph
        self.graph_map()

        self.performance = []
        self.performance.append(0)
        self.performance.append(0)

    def __call__(self):
        return self.H

    def n_cube(self, dim, symmetry=False, printout=False):
        """
        Generate the simplicial triangulation of the N-D hypercube
        containing 2**n vertices
        """
        origin = list(np.zeros(dim, dtype=int))
        self.origin = origin
        supremum = list(np.ones(dim, dtype=int))
        self.supremum = supremum

        # tuple versions for indexing
        origintuple = tuple(origin)
        supremumtuple = tuple(supremum)

        x_parents = [origintuple]

        if symmetry:
            self.C0 = Simplex(0, 0, 0, self.dim)  # Initial cell object
            self.C0.add_vertex(self.V[origintuple])

            i_s = 0
            self.perm_symmetry(i_s, x_parents, origin)
            self.C0.add_vertex(self.V[supremumtuple])
        else:
            self.C0 = Cell(0, 0, origin, supremum)  # Initial cell object
            self.C0.add_vertex(self.V[origintuple])
            self.C0.add_vertex(self.V[supremumtuple])

            i_parents = []
            self.perm(i_parents, x_parents, origin)

        if printout:
            print("Initial hyper cube:")
            for v in self.C0():
                v.print_out()

    def perm(self, i_parents, x_parents, xi):
        # TODO: Cut out of for if outside linear constraint cutting planes
        xi_t = tuple(xi)

        # Construct required iterator
        iter_range = [x for x in range(self.dim) if x not in i_parents]

        for i in iter_range:
            i2_parents = copy.copy(i_parents)
            i2_parents.append(i)
            xi2 = copy.copy(xi)
            xi2[i] = 1
            # Make new vertex list a hashable tuple
            xi2_t = tuple(xi2)
            # Append to cell
            self.C0.add_vertex(self.V[xi2_t])
            # Connect neighbors and vice versa
            # Parent point
            self.V[xi2_t].connect(self.V[xi_t])

            # Connect all family of simplices in parent containers
            for x_ip in x_parents:
                self.V[xi2_t].connect(self.V[x_ip])

            x_parents2 = copy.copy(x_parents)
            x_parents2.append(xi_t)

            # Permutate
            self.perm(i2_parents, x_parents2, xi2)

    def perm_symmetry(self, i_s, x_parents, xi):
        # TODO: Cut out of for if outside linear constraint cutting planes
        xi_t = tuple(xi)
        xi2 = copy.copy(xi)
        xi2[i_s] = 1
        # Make new vertex list a hashable tuple
        xi2_t = tuple(xi2)
        # Append to cell
        self.C0.add_vertex(self.V[xi2_t])
        # Connect neighbors and vice versa
        # Parent point
        self.V[xi2_t].connect(self.V[xi_t])

        # Connect all family of simplices in parent containers
        for x_ip in x_parents:
            self.V[xi2_t].connect(self.V[x_ip])

        x_parents2 = copy.copy(x_parents)
        x_parents2.append(xi_t)

        i_s += 1
        if i_s == self.dim:
            return
        # Permutate
        self.perm_symmetry(i_s, x_parents2, xi2)

    def add_centroid(self):
        """Split the central edge between the origin and supremum of
        a cell and add the new vertex to the complex"""
        self.centroid = list(
            (np.array(self.origin) + np.array(self.supremum)) / 2.0)
        self.C0.add_vertex(self.V[tuple(self.centroid)])
        self.C0.centroid = self.centroid

        # Disconnect origin and supremum
        self.V[tuple(self.origin)].disconnect(self.V[tuple(self.supremum)])

        # Connect centroid to all other vertices
        for v in self.C0():
            self.V[tuple(self.centroid)].connect(self.V[tuple(v.x)])

        self.centroid_added = True
        return

    # Construct incidence array:
    def incidence(self):
        if self.centroid_added:
            self.structure = np.zeros([2 ** self.dim + 1, 2 ** self.dim + 1],
                                         dtype=int)
        else:
            self.structure = np.zeros([2 ** self.dim, 2 ** self.dim],
                                         dtype=int)

        for v in self.HC.C0():
            for v2 in v.nn:
                self.structure[v.index, v2.index] = 1

        return

    # A more sparse incidence generator:
    def graph_map(self):
        """ Make a list of size 2**n + 1 where an entry is a vertex
        incidence, each list element contains a list of indexes
        corresponding to that entries neighbors"""

        self.graph = [[v2.index for v2 in v.nn] for v in self.C0()]

    # Graph structure method:
    # 0. Capture the indices of the initial cell.
    # 1. Generate new origin and supremum scalars based on current generation
    # 2. Generate a new set of vertices corresponding to a new
    #    "origin" and "supremum"
    # 3. Connected based on the indices of the previous graph structure
    # 4. Disconnect the edges in the original cell

    def sub_generate_cell(self, C_i, gen):
        """Subgenerate a cell `C_i` of generation `gen` and
        homology group rank `hgr`."""
        origin_new = tuple(C_i.centroid)
        centroid_index = len(C_i()) - 1

        # If not gen append
        try:
            self.H[gen]
        except IndexError:
            self.H.append([])

        # Generate subcubes using every extreme vertex in C_i as a supremum
        # and the centroid of C_i as the origin
        H_new = []  # list storing all the new cubes split from C_i
        for i, v in enumerate(C_i()[:-1]):
            supremum = tuple(v.x)
            H_new.append(
                self.construct_hypercube(origin_new, supremum, gen, C_i.hg_n))

        for i, connections in enumerate(self.graph):
            # Present vertex V_new[i]; connect to all connections:
            if i == centroid_index:  # Break out of centroid
                break

            for j in connections:
                C_i()[i].disconnect(C_i()[j])

        # Destroy the old cell
        if C_i is not self.C0:  # Garbage collector does this anyway; not needed
            del C_i

        # TODO: Recalculate all the homology group ranks of each cell
        return H_new

    def split_generation(self):
        """
        Run sub_generate_cell for every cell in the current complex self.gen
        """
        no_splits = False  # USED IN SHGO
        try:
            for c in self.H[self.gen]:
                if self.symmetry:
                    # self.sub_generate_cell_symmetry(c, self.gen + 1)
                    self.split_simplex_symmetry(c, self.gen + 1)
                else:
                    self.sub_generate_cell(c, self.gen + 1)
        except IndexError:
            no_splits = True  # USED IN SHGO

        self.gen += 1
        return no_splits  # USED IN SHGO

    def construct_hypercube(self, origin, supremum, gen, hgr,
                            printout=False):
        """
        Build a hypercube with triangulations symmetric to C0.

        Parameters
        ----------
        origin : vec
        supremum : vec (tuple)
        gen : generation
        hgr : parent homology group rank
        """
        # Initiate new cell
        v_o = np.array(origin)
        v_s = np.array(supremum)

        C_new = Cell(gen, hgr, origin, supremum)
        C_new.centroid = tuple((v_o + v_s) * .5)

        # Build new indexed vertex list
        V_new = []

        for i, v in enumerate(self.C0()[:-1]):
            v_x = np.array(v.x)
            sub_cell_t1 = v_o - v_o * v_x
            sub_cell_t2 = v_s * v_x

            vec = sub_cell_t1 + sub_cell_t2

            vec = tuple(vec)
            C_new.add_vertex(self.V[vec])
            V_new.append(vec)

        # Add new centroid
        C_new.add_vertex(self.V[C_new.centroid])
        V_new.append(C_new.centroid)

        # Connect new vertices #TODO: Thread into other loop; no need for V_new
        for i, connections in enumerate(self.graph):
            # Present vertex V_new[i]; connect to all connections:
            for j in connections:
                self.V[V_new[i]].connect(self.V[V_new[j]])

        if printout:
            print("A sub hyper cube with:")
            print("origin: {}".format(origin))
            print("supremum: {}".format(supremum))
            for v in C_new():
                v.print_out()

        # Append the new cell to the to complex
        self.H[gen].append(C_new)

        return C_new

    def split_simplex_symmetry(self, S, gen):
        """
        Split a hypersimplex S into two sub simplices by building a hyperplane
        which connects to a new vertex on an edge (the longest edge in
        dim = {2, 3}) and every other vertex in the simplex that is not
        connected to the edge being split.

        This function utilizes the knowledge that the problem is specified
        with symmetric constraints

        The longest edge is tracked by an ordering of the
        vertices in every simplices, the edge between first and second
        vertex is the longest edge to be split in the next iteration.
        """
        # If not gen append
        try:
            self.H[gen]
        except IndexError:
            self.H.append([])

        # Find new vertex.
        # V_new_x = tuple((np.array(C()[0].x) + np.array(C()[1].x)) / 2.0)
        s = S()
        firstx = s[0].x
        lastx = s[-1].x
        V_new = self.V[tuple((np.array(firstx) + np.array(lastx)) / 2.0)]

        # Disconnect old longest edge
        self.V[firstx].disconnect(self.V[lastx])

        # Connect new vertices to all other vertices
        for v in s[:]:
            v.connect(self.V[V_new.x])

        # New "lower" simplex
        S_new_l = Simplex(gen, S.hg_n, self.generation_cycle,
                          self.dim)
        S_new_l.add_vertex(s[0])
        S_new_l.add_vertex(V_new)  # Add new vertex
        for v in s[1:-1]:  # Add all other vertices
            S_new_l.add_vertex(v)

        # New "upper" simplex
        S_new_u = Simplex(gen, S.hg_n, S.generation_cycle, self.dim)

        # First vertex on new long edge
        S_new_u.add_vertex(s[S_new_u.generation_cycle + 1])

        for v in s[1:-1]:  # Remaining vertices
            S_new_u.add_vertex(v)

        for k, v in enumerate(s[1:-1]):  # iterate through inner vertices
            if k == S.generation_cycle:
                S_new_u.add_vertex(V_new)
            else:
                S_new_u.add_vertex(v)

        S_new_u.add_vertex(s[-1])  # Second vertex on new long edge

        self.H[gen].append(S_new_l)
        self.H[gen].append(S_new_u)

        return

    # Plots
    def plot_complex(self):
        """
             Here, C is the LIST of simplexes S in the
             2- or 3-D complex

             To plot a single simplex S in a set C, use e.g., [C[0]]
        """
        from matplotlib import pyplot  # type: ignore[import]
        if self.dim == 2:
            pyplot.figure()
            for C in self.H:
                for c in C:
                    for v in c():
                        if self.bounds is None:
                            x_a = np.array(v.x, dtype=float)
                        else:
                            x_a = np.array(v.x, dtype=float)
                            for i in range(len(self.bounds)):
                                x_a[i] = (x_a[i] * (self.bounds[i][1]
                                                    - self.bounds[i][0])
                                          + self.bounds[i][0])

                        # logging.info('v.x_a = {}'.format(x_a))

                        pyplot.plot([x_a[0]], [x_a[1]], 'o')

                        xlines = []
                        ylines = []
                        for vn in v.nn:
                            if self.bounds is None:
                                xn_a = np.array(vn.x, dtype=float)
                            else:
                                xn_a = np.array(vn.x, dtype=float)
                                for i in range(len(self.bounds)):
                                    xn_a[i] = (xn_a[i] * (self.bounds[i][1]
                                                          - self.bounds[i][0])
                                               + self.bounds[i][0])

                            # logging.info('vn.x = {}'.format(vn.x))

                            xlines.append(xn_a[0])
                            ylines.append(xn_a[1])
                            xlines.append(x_a[0])
                            ylines.append(x_a[1])

                        pyplot.plot(xlines, ylines)

            if self.bounds is None:
                pyplot.ylim([-1e-2, 1 + 1e-2])
                pyplot.xlim([-1e-2, 1 + 1e-2])
            else:
                pyplot.ylim(
                    [self.bounds[1][0] - 1e-2, self.bounds[1][1] + 1e-2])
                pyplot.xlim(
                    [self.bounds[0][0] - 1e-2, self.bounds[0][1] + 1e-2])

            pyplot.show()

        elif self.dim == 3:
            fig = pyplot.figure()
            ax = fig.add_subplot(111, projection='3d')

            for C in self.H:
                for c in C:
                    for v in c():
                        x = []
                        y = []
                        z = []
                        # logging.info('v.x = {}'.format(v.x))
                        x.append(v.x[0])
                        y.append(v.x[1])
                        z.append(v.x[2])
                        for vn in v.nn:
                            x.append(vn.x[0])
                            y.append(vn.x[1])
                            z.append(vn.x[2])
                            x.append(v.x[0])
                            y.append(v.x[1])
                            z.append(v.x[2])
                            # logging.info('vn.x = {}'.format(vn.x))

                        ax.plot(x, y, z, label='simplex')

            pyplot.show()
        else:
            print("dimension higher than 3 or wrong complex format")
        return


class VertexGroup:
    def __init__(self, p_gen, p_hgr):
        self.p_gen = p_gen  # parent generation
        self.p_hgr = p_hgr  # parent homology group rank
        self.hg_n = None
        self.hg_d = None

        # Maybe add parent homology group rank total history
        # This is the sum off all previously split cells
        # cumulatively throughout its entire history
        self.C = []

    def __call__(self):
        return self.C

    def add_vertex(self, V):
        if V not in self.C:
            self.C.append(V)

    def homology_group_rank(self):
        """
        Returns the homology group order of the current cell
        """
        if self.hg_n is None:
            self.hg_n = sum(1 for v in self.C if v.minimiser())

        return self.hg_n

    def homology_group_differential(self):
        """
        Returns the difference between the current homology group of the
        cell and its parent group
        """
        if self.hg_d is None:
            self.hgd = self.hg_n - self.p_hgr

        return self.hgd

    def polytopial_sperner_lemma(self):
        """
        Returns the number of stationary points theoretically contained in the
        cell based information currently known about the cell
        """
        pass

    def print_out(self):
        """
        Print the current cell to console
        """
        for v in self():
            v.print_out()


class Cell(VertexGroup):
    """
    Contains a cell that is symmetric to the initial hypercube triangulation
    """

    def __init__(self, p_gen, p_hgr, origin, supremum):
        super().__init__(p_gen, p_hgr)

        self.origin = origin
        self.supremum = supremum
        self.centroid = None  # (Not always used)
        # TODO: self.bounds


class Simplex(VertexGroup):
    """
    Contains a simplex that is symmetric to the initial symmetry constrained
    hypersimplex triangulation
    """

    def __init__(self, p_gen, p_hgr, generation_cycle, dim):
        super().__init__(p_gen, p_hgr)

        self.generation_cycle = (generation_cycle + 1) % (dim - 1)


class Vertex:
    def __init__(self, x, bounds=None, func=None, func_args=(), g_cons=None,
                 g_cons_args=(), nn=None, index=None):
        self.x = x
        self.order = sum(x)
        x_a = np.array(x, dtype=float)
        if bounds is not None:
            for i, (lb, ub) in enumerate(bounds):
                x_a[i] = x_a[i] * (ub - lb) + lb

        # TODO: Make saving the array structure optional
        self.x_a = x_a

        # Note Vertex is only initiated once for all x so only
        # evaluated once
        if func is not None:
            self.feasible = True
            if g_cons is not None:
                for g, args in zip(g_cons, g_cons_args):
                    if g(self.x_a, *args) < 0.0:
                        self.f = np.inf
                        self.feasible = False
                        break
            if self.feasible:
                self.f = func(x_a, *func_args)

        if nn is not None:
            self.nn = nn
        else:
            self.nn = set()

        self.fval = None
        self.check_min = True

        # Index:
        if index is not None:
            self.index = index

    def __hash__(self):
        return hash(self.x)

    def connect(self, v):
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

            if self.minimiser():
                v._min = False
                v.check_min = False

            # TEMPORARY
            self.check_min = True
            v.check_min = True

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)
            self.check_min = True
            v.check_min = True

    def minimiser(self):
        """Check whether this vertex is strictly less than all its neighbors"""
        if self.check_min:
            self._min = all(self.f < v.f for v in self.nn)
            self.check_min = False

        return self._min

    def print_out(self):
        print("Vertex: {}".format(self.x))
        constr = 'Connections: '
        for vc in self.nn:
            constr += '{} '.format(vc.x)

        print(constr)
        print('Order = {}'.format(self.order))


class VertexCache:
    def __init__(self, func, func_args=(), bounds=None, g_cons=None,
                 g_cons_args=(), indexed=True):

        self.cache = {}
        self.func = func
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args
        self.func_args = func_args
        self.bounds = bounds
        self.nfev = 0
        self.size = 0

        if indexed:
            self.index = -1

    def __getitem__(self, x, indexed=True):
        try:
            return self.cache[x]
        except KeyError:
            if indexed:
                self.index += 1
                xval = Vertex(x, bounds=self.bounds,
                              func=self.func, func_args=self.func_args,
                              g_cons=self.g_cons,
                              g_cons_args=self.g_cons_args,
                              index=self.index)
            else:
                xval = Vertex(x, bounds=self.bounds,
                              func=self.func, func_args=self.func_args,
                              g_cons=self.g_cons,
                              g_cons_args=self.g_cons_args)

            # logging.info("New generated vertex at x = {}".format(x))
            # NOTE: Surprisingly high performance increase if logging is commented out
            self.cache[x] = xval

            # TODO: Check
            if self.func is not None:
                if self.g_cons is not None:
                    if xval.feasible:
                        self.nfev += 1
                        self.size += 1
                    else:
                        self.size += 1
                else:
                    self.nfev += 1
                    self.size += 1

            return self.cache[x]
