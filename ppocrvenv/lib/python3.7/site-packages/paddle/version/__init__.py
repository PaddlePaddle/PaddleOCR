# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
full_version    = '2.2.2'
major           = '2'
minor           = '2'
patch           = '2'
rc              = '0'
cuda_version    = 'False'
cudnn_version   = 'False'
istaged         = True
commit          = 'b031c389938bfa15e15bb20494c76f86289d77b0'
with_mkl        = 'OFF'

__all__ = ['cuda', 'cudnn']

def show():
    if istaged:
        print('full_version:', full_version)
        print('major:', major)
        print('minor:', minor)
        print('patch:', patch)
        print('rc:', rc)
    else:
        print('commit:', commit)

def mkl():
    return with_mkl

def cuda():
    """Get cuda version of paddle package.

    Returns:
        string: Return the version information of cuda. If paddle package is CPU version, it will return False.
    
    Examples:
        .. code-block:: python

            import paddle

            paddle.version.cuda()
            # '10.2'

    """
    return cuda_version

def cudnn():
    """Get cudnn version of paddle package.

    Returns:
        string: Return the version information of cudnn. If paddle package is CPU version, it will return False.
    
    Examples:
        .. code-block:: python

            import paddle

            paddle.version.cudnn()
            # '7.6.5'

    """
    return cudnn_version
