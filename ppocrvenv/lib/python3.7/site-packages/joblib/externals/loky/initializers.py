import warnings


def _viztracer_init(init_kwargs):
    """Initialize viztracer's profiler in worker processes"""
    from viztracer import VizTracer
    tracer = VizTracer(**init_kwargs)
    tracer.register_exit()
    tracer.start()


def _make_viztracer_initializer_and_initargs():
    try:
        import viztracer
        tracer = viztracer.get_tracer()
        if tracer is not None and getattr(tracer, 'enable', False):
            # Profiler is active: introspect its configuration to
            # initialize the workers with the same configuration.
            return _viztracer_init, (tracer.init_kwargs,)
    except ImportError:
        # viztracer is not installed: nothing to do
        pass
    except Exception as e:
        # In case viztracer's API evolve, we do not want to crash loky but
        # we want to know about it to be able to update loky.
        warnings.warn("Unable to introspect viztracer state: {}"
                      .format(e))
    return None, ()


class _ChainedInitializer():
    """Compound worker initializer

    This is meant to be used in conjunction with _chain_initializers to
    produce  the necessary chained_args list to be passed to __call__.
    """

    def __init__(self, initializers):
        self._initializers = initializers

    def __call__(self, *chained_args):
        for initializer, args in zip(self._initializers, chained_args):
            initializer(*args)


def _chain_initializers(initializer_and_args):
    """Convenience helper to combine a sequence of initializers.

    If some initializers are None, they are filtered out.
    """
    filtered_initializers = []
    filtered_initargs = []
    for initializer, initargs in initializer_and_args:
        if initializer is not None:
            filtered_initializers.append(initializer)
            filtered_initargs.append(initargs)

    if len(filtered_initializers) == 0:
        return None, ()
    elif len(filtered_initializers) == 1:
        return filtered_initializers[0], filtered_initargs[0]
    else:
        return _ChainedInitializer(filtered_initializers), filtered_initargs


def _prepare_initializer(initializer, initargs):
    if initializer is not None and not callable(initializer):
        raise TypeError(
            "initializer must be a callable, got: {!r}"
            .format(initializer)
        )

    # Introspect runtime to determine if we need to propagate the viztracer
    # profiler information to the workers:
    return _chain_initializers([
        (initializer, initargs),
        _make_viztracer_initializer_and_initargs(),
    ])
