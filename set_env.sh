#. /paddle/set_env.sh↩
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export FLAGS_fraction_of_gpu_memory_to_use=1.0
python_bin_dir="/opt/_internal/cpython-3.7.0/bin/"
alias python=$python_bin_dir"python3.7"
alias pip=$python_bin_dir"pip3.7"
alias ipython=$python_bin_dir"ipython3"
export  LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:.
ldconfig
