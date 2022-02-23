"""
=====================================================
Optimization and root finding (:mod:`scipy.optimize`)
=====================================================

.. currentmodule:: scipy.optimize

SciPy ``optimize`` provides functions for minimizing (or maximizing)
objective functions, possibly subject to constraints. It includes
solvers for nonlinear problems (with support for both local and global
optimization algorithms), linear programing, constrained
and nonlinear least-squares, root finding, and curve fitting.

Common functions and objects, shared across different solvers, are:

.. autosummary::
   :toctree: generated/

   show_options - Show specific options optimization solvers.
   OptimizeResult - The optimization result returned by some optimizers.
   OptimizeWarning - The optimization encountered problems.


Optimization
============

Scalar functions optimization
-----------------------------

.. autosummary::
   :toctree: generated/

   minimize_scalar - Interface for minimizers of univariate functions

The `minimize_scalar` function supports the following methods:

.. toctree::

   optimize.minimize_scalar-brent
   optimize.minimize_scalar-bounded
   optimize.minimize_scalar-golden

Local (multivariate) optimization
---------------------------------

.. autosummary::
   :toctree: generated/

   minimize - Interface for minimizers of multivariate functions.

The `minimize` function supports the following methods:

.. toctree::

   optimize.minimize-neldermead
   optimize.minimize-powell
   optimize.minimize-cg
   optimize.minimize-bfgs
   optimize.minimize-newtoncg
   optimize.minimize-lbfgsb
   optimize.minimize-tnc
   optimize.minimize-cobyla
   optimize.minimize-slsqp
   optimize.minimize-trustconstr
   optimize.minimize-dogleg
   optimize.minimize-trustncg
   optimize.minimize-trustkrylov
   optimize.minimize-trustexact

Constraints are passed to `minimize` function as a single object or
as a list of objects from the following classes:

.. autosummary::
   :toctree: generated/

   NonlinearConstraint - Class defining general nonlinear constraints.
   LinearConstraint - Class defining general linear constraints.

Simple bound constraints are handled separately and there is a special class
for them:

.. autosummary::
   :toctree: generated/

   Bounds - Bound constraints.

Quasi-Newton strategies implementing `HessianUpdateStrategy`
interface can be used to approximate the Hessian in `minimize`
function (available only for the 'trust-constr' method). Available
quasi-Newton methods implementing this interface are:

.. autosummary::
   :toctree: generated/

   BFGS - Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update strategy.
   SR1 - Symmetric-rank-1 Hessian update strategy.

Global optimization
-------------------

.. autosummary::
   :toctree: generated/

   basinhopping - Basinhopping stochastic optimizer.
   brute - Brute force searching optimizer.
   differential_evolution - stochastic minimization using differential evolution.

   shgo - simplicial homology global optimisation
   dual_annealing - Dual annealing stochastic optimizer.


Least-squares and curve fitting
===============================

Nonlinear least-squares
-----------------------

.. autosummary::
   :toctree: generated/

   least_squares - Solve a nonlinear least-squares problem with bounds on the variables.

Linear least-squares
--------------------

.. autosummary::
   :toctree: generated/

   nnls - Linear least-squares problem with non-negativity constraint.
   lsq_linear - Linear least-squares problem with bound constraints.

Curve fitting
-------------

.. autosummary::
   :toctree: generated/

   curve_fit -- Fit curve to a set of points.

Root finding
============

Scalar functions
----------------
.. autosummary::
   :toctree: generated/

   root_scalar - Unified interface for nonlinear solvers of scalar functions.
   brentq - quadratic interpolation Brent method.
   brenth - Brent method, modified by Harris with hyperbolic extrapolation.
   ridder - Ridder's method.
   bisect - Bisection method.
   newton - Newton's method (also Secant and Halley's methods).
   toms748 - Alefeld, Potra & Shi Algorithm 748.
   RootResults - The root finding result returned by some root finders.

The `root_scalar` function supports the following methods:

.. toctree::

   optimize.root_scalar-brentq
   optimize.root_scalar-brenth
   optimize.root_scalar-bisect
   optimize.root_scalar-ridder
   optimize.root_scalar-newton
   optimize.root_scalar-toms748
   optimize.root_scalar-secant
   optimize.root_scalar-halley



The table below lists situations and appropriate methods, along with
*asymptotic* convergence rates per iteration (and per function evaluation)
for successful convergence to a simple root(*).
Bisection is the slowest of them all, adding one bit of accuracy for each
function evaluation, but is guaranteed to converge.
The other bracketing methods all (eventually) increase the number of accurate
bits by about 50% for every function evaluation.
The derivative-based methods, all built on `newton`, can converge quite quickly
if the initial value is close to the root.  They can also be applied to
functions defined on (a subset of) the complex plane.

+-------------+----------+----------+-----------+-------------+-------------+----------------+
| Domain of f | Bracket? |    Derivatives?      | Solvers     |        Convergence           |
+             +          +----------+-----------+             +-------------+----------------+
|             |          | `fprime` | `fprime2` |             | Guaranteed? |  Rate(s)(*)    |
+=============+==========+==========+===========+=============+=============+================+
| `R`         | Yes      | N/A      | N/A       | - bisection | - Yes       | - 1 "Linear"   |
|             |          |          |           | - brentq    | - Yes       | - >=1, <= 1.62 |
|             |          |          |           | - brenth    | - Yes       | - >=1, <= 1.62 |
|             |          |          |           | - ridder    | - Yes       | - 2.0 (1.41)   |
|             |          |          |           | - toms748   | - Yes       | - 2.7 (1.65)   |
+-------------+----------+----------+-----------+-------------+-------------+----------------+
| `R` or `C`  | No       | No       | No        | secant      | No          | 1.62 (1.62)    |
+-------------+----------+----------+-----------+-------------+-------------+----------------+
| `R` or `C`  | No       | Yes      | No        | newton      | No          | 2.00 (1.41)    |
+-------------+----------+----------+-----------+-------------+-------------+----------------+
| `R` or `C`  | No       | Yes      | Yes       | halley      | No          | 3.00 (1.44)    |
+-------------+----------+----------+-----------+-------------+-------------+----------------+

.. seealso::

   `scipy.optimize.cython_optimize` -- Typed Cython versions of zeros functions

Fixed point finding:

.. autosummary::
   :toctree: generated/

   fixed_point - Single-variable fixed-point solver.

Multidimensional
----------------

.. autosummary::
   :toctree: generated/

   root - Unified interface for nonlinear solvers of multivariate functions.

The `root` function supports the following methods:

.. toctree::

   optimize.root-hybr
   optimize.root-lm
   optimize.root-broyden1
   optimize.root-broyden2
   optimize.root-anderson
   optimize.root-linearmixing
   optimize.root-diagbroyden
   optimize.root-excitingmixing
   optimize.root-krylov
   optimize.root-dfsane

Linear programming
==================

.. autosummary::
   :toctree: generated/

   linprog -- Unified interface for minimizers of linear programming problems.

The `linprog` function supports the following methods:

.. toctree::

   optimize.linprog-simplex
   optimize.linprog-interior-point
   optimize.linprog-revised_simplex
   optimize.linprog-highs-ipm
   optimize.linprog-highs-ds
   optimize.linprog-highs

The simplex, interior-point, and revised simplex methods support callback
functions, such as:

.. autosummary::
   :toctree: generated/

   linprog_verbose_callback -- Sample callback function for linprog (simplex).

Assignment problems
===================

.. autosummary::
   :toctree: generated/

   linear_sum_assignment -- Solves the linear-sum assignment problem.
   quadratic_assignment -- Solves the quadratic assignment problem.

The `quadratic_assignment` function supports the following methods:

.. toctree::

   optimize.qap-faq
   optimize.qap-2opt

Utilities
=========

Finite-difference approximation
-------------------------------

.. autosummary::
   :toctree: generated/

   approx_fprime - Approximate the gradient of a scalar function.
   check_grad - Check the supplied derivative using finite differences.


Line search
-----------

.. autosummary::
   :toctree: generated/

   bracket - Bracket a minimum, given two starting points.
   line_search - Return a step that satisfies the strong Wolfe conditions.

Hessian approximation
---------------------

.. autosummary::
   :toctree: generated/

   LbfgsInvHessProduct - Linear operator for L-BFGS approximate inverse Hessian.
   HessianUpdateStrategy - Interface for implementing Hessian update strategies

Benchmark problems
------------------

.. autosummary::
   :toctree: generated/

   rosen - The Rosenbrock function.
   rosen_der - The derivative of the Rosenbrock function.
   rosen_hess - The Hessian matrix of the Rosenbrock function.
   rosen_hess_prod - Product of the Rosenbrock Hessian with a vector.

Legacy functions
================

The functions below are not recommended for use in new scripts;
all of these methods are accessible via a newer, more consistent
interfaces, provided by the interfaces above.

Optimization
------------

General-purpose multivariate methods:

.. autosummary::
   :toctree: generated/

   fmin - Nelder-Mead Simplex algorithm.
   fmin_powell - Powell's (modified) level set method.
   fmin_cg - Non-linear (Polak-Ribiere) conjugate gradient algorithm.
   fmin_bfgs - Quasi-Newton method (Broydon-Fletcher-Goldfarb-Shanno).
   fmin_ncg - Line-search Newton Conjugate Gradient.

Constrained multivariate methods:

.. autosummary::
   :toctree: generated/

   fmin_l_bfgs_b - Zhu, Byrd, and Nocedal's constrained optimizer.
   fmin_tnc - Truncated Newton code.
   fmin_cobyla - Constrained optimization by linear approximation.
   fmin_slsqp - Minimization using sequential least-squares programming.

Univariate (scalar) minimization methods:

.. autosummary::
   :toctree: generated/

   fminbound - Bounded minimization of a scalar function.
   brent - 1-D function minimization using Brent method.
   golden - 1-D function minimization using Golden Section method.

Least-squares
-------------

.. autosummary::
   :toctree: generated/

   leastsq - Minimize the sum of squares of M equations in N unknowns.

Root finding
------------

General nonlinear solvers:

.. autosummary::
   :toctree: generated/

   fsolve - Non-linear multivariable equation solver.
   broyden1 - Broyden's first method.
   broyden2 - Broyden's second method.

Large-scale nonlinear solvers:

.. autosummary::
   :toctree: generated/

   newton_krylov
   anderson

Simple iteration solvers:

.. autosummary::
   :toctree: generated/

   excitingmixing
   linearmixing
   diagbroyden

:mod:`Additional information on the nonlinear solvers <scipy.optimize.nonlin>`
"""

from .optimize import *
from ._minimize import *
from ._root import *
from ._root_scalar import *
from .minpack import *
from .zeros import *
from .lbfgsb import fmin_l_bfgs_b, LbfgsInvHessProduct
from .tnc import fmin_tnc
from .cobyla import fmin_cobyla
from .nonlin import *
from .slsqp import fmin_slsqp
from ._nnls import nnls
from ._basinhopping import basinhopping
from ._linprog import linprog, linprog_verbose_callback
from ._lsap import linear_sum_assignment
from ._differentialevolution import differential_evolution
from ._lsq import least_squares, lsq_linear
from ._constraints import (NonlinearConstraint,
                           LinearConstraint,
                           Bounds)
from ._hessian_update_strategy import HessianUpdateStrategy, BFGS, SR1
from ._shgo import shgo
from ._dual_annealing import dual_annealing
from ._qap import quadratic_assignment

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
