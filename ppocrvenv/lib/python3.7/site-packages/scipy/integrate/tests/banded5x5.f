c   banded5x5.f
c
c   This Fortran library contains implementations of the
c   differential equation
c       dy/dt = A*y
c   where A is a 5x5 banded matrix (see below for the actual
c   values).  These functions will be used to test
c   scipy.integrate.odeint.
c
c   The idea is to solve the system two ways: pure Fortran, and
c   using odeint.  The "pure Fortran" solver is implemented in
c   the subroutine banded5x5_solve below.  It calls LSODA to
c   solve the system.
c
c   To solve the same system using odeint, the functions in this
c   file are given a python wrapper using f2py.  Then the code
c   in test_odeint_jac.py uses the wrapper to implement the
c   equation and Jacobian functions required by odeint.  Because
c   those functions ultimately call the Fortran routines defined
c   in this file, the two method (pure Fortran and odeint) should
c   produce exactly the same results.  (That's assuming floating
c   point calculations are deterministic, which can be an
c   incorrect assumption.)  If we simply re-implemented the
c   equation and Jacobian functions using just python and numpy,
c   the floating point calculations would not be performed in
c   the same sequence as in the Fortran code, and we would obtain
c   different answers.  The answer for either method would be
c   numerically "correct", but the errors would be different,
c   and the counts of function and Jacobian evaluations would
c   likely be different.
c
      block data jacobian
      implicit none

      double precision bands
      dimension bands(4,5)
      common /jac/ bands

c     The data for a banded Jacobian stored in packed banded
c     format.  The full Jacobian is
c
c           -1,  0.25,     0,     0,     0
c         0.25,    -5,  0.25,     0,     0
c         0.10,  0.25,   -25,  0.25,     0
c            0,  0.10,  0.25,  -125,  0.25
c            0,     0,  0.10,  0.25,  -625
c
c     The columns in the following layout of numbers are
c     the upper diagonal, main diagonal and two lower diagonals
c     (i.e. each row in the layout is a column of the packed
c     banded Jacobian).  The values 0.00D0 are in the "don't
c     care" positions.

      data bands/
     +      0.00D0,   -1.0D0, 0.25D0, 0.10D0,
     +      0.25D0,   -5.0D0, 0.25D0, 0.10D0,
     +      0.25D0,  -25.0D0, 0.25D0, 0.10D0,
     +      0.25D0, -125.0D0, 0.25D0, 0.00D0,
     +      0.25D0, -625.0D0, 0.00D0, 0.00D0
     +      /

      end

      subroutine getbands(jac)
      double precision jac
      dimension jac(4, 5)
cf2py intent(out) jac

      double precision bands
      dimension bands(4,5)
      common /jac/ bands

      integer i, j
      do 5 i = 1, 4
          do 5 j = 1, 5
              jac(i, j) = bands(i, j)
 5    continue

      return
      end

c
c Differential equations, right-hand-side
c
      subroutine banded5x5(n, t, y, f)
      implicit none
      integer n
      double precision t, y, f
      dimension y(n), f(n)

      double precision bands
      dimension bands(4,5)
      common /jac/ bands

      f(1) = bands(2,1)*y(1) + bands(1,2)*y(2)
      f(2) = bands(3,1)*y(1) + bands(2,2)*y(2) + bands(1,3)*y(3)
      f(3) = bands(4,1)*y(1) + bands(3,2)*y(2) + bands(2,3)*y(3)
     +                                         + bands(1,4)*y(4)
      f(4) = bands(4,2)*y(2) + bands(3,3)*y(3) + bands(2,4)*y(4)
     +                                         + bands(1,5)*y(5)
      f(5) = bands(4,3)*y(3) + bands(3,4)*y(4) + bands(2,5)*y(5)

      return
      end

c
c Jacobian
c
c The subroutine assumes that the full Jacobian is to be computed.
c ml and mu are ignored, and nrowpd is assumed to be n.
c
      subroutine banded5x5_jac(n, t, y, ml, mu, jac, nrowpd)
      implicit none
      integer n, ml, mu, nrowpd
      double precision t, y, jac
      dimension y(n), jac(nrowpd, n)

      integer i, j

      double precision bands
      dimension bands(4,5)
      common /jac/ bands

      do 15 i = 1, 4
          do 15 j = 1, 5
              if ((i - j) .gt. 0) then
                  jac(i - j, j) = bands(i, j)
              end if
15    continue

      return
      end

c
c Banded Jacobian
c
c ml = 2, mu = 1
c
      subroutine banded5x5_bjac(n, t, y, ml, mu, bjac, nrowpd)
      implicit none
      integer n, ml, mu, nrowpd
      double precision t, y, bjac
      dimension y(5), bjac(nrowpd, n)

      integer i, j

      double precision bands
      dimension bands(4,5)
      common /jac/ bands

      do 20 i = 1, 4
          do 20 j = 1, 5
              bjac(i, j) = bands(i, j)
 20   continue

      return
      end


      subroutine banded5x5_solve(y, nsteps, dt, jt, nst, nfe, nje)

c     jt is the Jacobian type:
c         jt = 1  Use the full Jacobian.
c         jt = 4  Use the banded Jacobian.
c     nst, nfe and nje are outputs:
c         nst:  Total number of internal steps
c         nfe:  Total number of function (i.e. right-hand-side)
c               evaluations
c         nje:  Total number of Jacobian evaluations

      implicit none

      external banded5x5
      external banded5x5_jac
      external banded5x5_bjac
      external LSODA

c     Arguments...
      double precision y, dt
      integer nsteps, jt, nst, nfe, nje
cf2py intent(inout) y
cf2py intent(in) nsteps, dt, jt
cf2py intent(out) nst, nfe, nje

c     Local variables...
      double precision atol, rtol, t, tout, rwork
      integer iwork
      dimension y(5), rwork(500), iwork(500)
      integer neq, i
      integer itol, iopt, itask, istate, lrw, liw

c     Common block...
      double precision jacband
      dimension jacband(4,5)
      common /jac/ jacband

c     --- t range ---
      t = 0.0D0

c     --- Solver tolerances ---
      rtol = 1.0D-11
      atol = 1.0D-13
      itol = 1

c     --- Other LSODA parameters ---
      neq = 5
      itask = 1
      istate = 1
      iopt = 0
      iwork(1) = 2
      iwork(2) = 1
      lrw = 500
      liw = 500

c     --- Call LSODA in a loop to compute the solution ---
      do 40 i = 1, nsteps
          tout = i*dt
          if (jt .eq. 1) then
              call LSODA(banded5x5, neq, y, t, tout,
     &               itol, rtol, atol, itask, istate, iopt,
     &               rwork, lrw, iwork, liw,
     &               banded5x5_jac, jt)
          else
              call LSODA(banded5x5, neq, y, t, tout,
     &               itol, rtol, atol, itask, istate, iopt,
     &               rwork, lrw, iwork, liw,
     &               banded5x5_bjac, jt)
          end if
 40       if (istate .lt. 0) goto 80

      nst = iwork(11)
      nfe = iwork(12)
      nje = iwork(13)

      return

 80   write (6,89) istate
 89   format(1X,"Error: istate=",I3)
      return
      end
