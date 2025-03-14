from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from pybind11.setup_helpers import Pybind11Extension
import sys

import os
#os.system("export CUSPARSELT_PATH=/home/roberto.lopez/Descargas/libcusparse_lt-linux-x86_64-0.3.0.3-archive")
os.system("export CUSPARSELT_PATH=/media/rtx3090/Disco2TB/inno4scale_shared/repositorios/libcusparse_lt-linux-x86_64-0.3.0.3-archive")

setup(
    name='spatha_sddmm',
    version='0.0.1',
    description='Custom library for Sparse Tensor Cores',
    author='Roberto L. Castro',
    author_email='roberto.lopez.castro@udc.es',
    ext_modules=[
            CUDAExtension('spatha_sddmm',
                              ['module_files/block_sparse/api/spatha.cu'],
                              extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_86', '--ptxas-options=-v', '-lineinfo']},
                              extra_link_args=['-lcudart','-lcusparse'])
                  ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)
