from numpy import add_newdoc

add_newdoc('scipy.odr', 'odr',
    """
    odr(fcn, beta0, y, x, we=None, wd=None, fjacb=None, fjacd=None, extra_args=None, ifixx=None, ifixb=None, job=0, iprint=0, errfile=None, rptfile=None, ndigit=0, taufac=0.0, sstol=-1.0, partol=-1.0, maxit=-1, stpb=None, stpd=None, sclb=None, scld=None, work=None, iwork=None, full_output=0)

    Low-level function for ODR.

    See Also
    --------
    ODR : The ODR class gathers all information and coordinates the running of the main fitting routine.
    Model : The Model class stores information about the function you wish to fit.
    Data : The data to fit.
    RealData : Data with weights as actual std. dev.s and/or covariances.

    Notes
    -----
    This is a function performing the same operation as the `ODR`,
    `Model`, and `Data` classes together. The parameters of this
    function are explained in the class documentation.

    """)

add_newdoc('scipy.odr.__odrpack', '_set_exceptions',
    """
    _set_exceptions(odr_error, odr_stop)

    Internal function: set exception classes.

    """)
