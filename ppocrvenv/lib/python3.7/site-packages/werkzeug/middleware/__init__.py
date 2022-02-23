"""
Middleware
==========

A WSGI middleware is a WSGI application that wraps another application
in order to observe or change its behavior. Werkzeug provides some
middleware for common use cases.

.. toctree::
    :maxdepth: 1

    proxy_fix
    shared_data
    dispatcher
    http_proxy
    lint
    profiler

The :doc:`interactive debugger </debug>` is also a middleware that can
be applied manually, although it is typically used automatically with
the :doc:`development server </serving>`.
"""
