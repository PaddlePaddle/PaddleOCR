
def pre_build_hook(build_ext, ext):
    from scipy._build_utils.compiler_helper import (
        set_cxx_flags_hook, try_add_flag)
    cc = build_ext._cxx_compiler
    args = ext.extra_compile_args

    set_cxx_flags_hook(build_ext, ext)

    if cc.compiler_type == 'msvc':
        args.append('/EHsc')
    else:
        try_add_flag(args, cc, '-fvisibility=hidden')


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('_uarray', parent_package, top_path)
    config.add_data_files('LICENSE')
    ext = config.add_extension('_uarray',
                               sources=['_uarray_dispatch.cxx'],
                               language='c++')
    ext._pre_build_hook = pre_build_hook
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
