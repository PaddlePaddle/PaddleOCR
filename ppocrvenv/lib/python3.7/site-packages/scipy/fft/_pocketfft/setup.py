
def pre_build_hook(build_ext, ext):
    from scipy._build_utils.compiler_helper import (
        set_cxx_flags_hook, try_add_flag, try_compile, has_flag)
    cc = build_ext._cxx_compiler
    args = ext.extra_compile_args

    set_cxx_flags_hook(build_ext, ext)

    if cc.compiler_type == 'msvc':
        args.append('/EHsc')
    else:
        # Use pthreads if available
        has_pthreads = try_compile(cc, code='#include <pthread.h>\n'
                                   'int main(int argc, char **argv) {}')
        if has_pthreads:
            ext.define_macros.append(('POCKETFFT_PTHREADS', None))
            if has_flag(cc, '-pthread'):
                args.append('-pthread')
                ext.extra_link_args.append('-pthread')
            else:
                raise RuntimeError("Build failed: System has pthreads header "
                                   "but could not compile with -pthread option")

        # Don't export library symbols
        try_add_flag(args, cc, '-fvisibility=hidden')


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import pybind11
    include_dirs = [pybind11.get_include(True), pybind11.get_include(False)]

    config = Configuration('_pocketfft', parent_package, top_path)
    ext = config.add_extension('pypocketfft',
                               sources=['pypocketfft.cxx'],
                               depends=['pocketfft_hdronly.h'],
                               include_dirs=include_dirs,
                               language='c++')
    ext._pre_build_hook = pre_build_hook

    config.add_data_files('LICENSE.md')
    config.add_data_dir('tests')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
