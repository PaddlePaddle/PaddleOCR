import numpy as np
from scipy.optimize import minimize, Bounds

def test_gh10880():
    # checks that verbose reporting works with trust-constr for
    # bound-contrained problems
    bnds = Bounds(1, 2)
    opts = {'maxiter': 1000, 'verbose': 2}
    minimize(lambda x: x**2, x0=2., method='trust-constr',
             bounds=bnds, options=opts)

    opts = {'maxiter': 1000, 'verbose': 3}
    minimize(lambda x: x**2, x0=2., method='trust-constr',
             bounds=bnds, options=opts)

def test_gh12922():
    # checks that verbose reporting works with trust-constr for
    # general constraints
    def objective(x):
        return np.array([(np.sum((x+1)**2))])

    cons = {'type': 'ineq', 'fun': lambda x: -x[0]**2}
    n = 25
    x0 = np.linspace(-5, 5, n)

    opts = {'maxiter': 1000, 'verbose': 2}
    result = minimize(objective, x0=x0, method='trust-constr',
                      constraints=cons, options=opts)

    opts = {'maxiter': 1000, 'verbose': 3}
    result = minimize(objective, x0=x0, method='trust-constr',
                      constraints=cons, options=opts)
