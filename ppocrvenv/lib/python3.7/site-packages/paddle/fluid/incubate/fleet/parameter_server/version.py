from __future__ import print_function

# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY

from paddle.fluid.incubate.fleet.base.mode import Mode

BUILD_MODE=Mode.TRANSPILER

def is_transpiler():
    return Mode.TRANSPILER == BUILD_MODE

