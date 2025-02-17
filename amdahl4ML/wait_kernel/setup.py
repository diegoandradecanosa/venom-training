from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from pybind11.setup_helpers import Pybind11Extension
import sys

setup(
    name='wait_kernels',
    version='0.0.1',
    description='Custom library for controlled wait cuda kernels',
    author='Diego Teijeiro Paredes',
    author_email='diego.teijeiro@udc.es',
    ext_modules=[
            CUDAExtension('wait_kernels',
                              ['controlled_wait_kernel.cu'],
                              extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_86', '--ptxas-options=-v', '-lineinfo']})
                  ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)