#!/bin/bash

module load CUDA
module load Python/3.10

python -m venv python_enviroment
source python_enviroment/bin/activate

pip install torch==2.0.1
pip install transformers==4.23.1 huggingface-hub==0.26.2 datasets==2.18.0

cd ../sparseml
pip install -e .[dev]
cd ../performance_analysis

pip install sten
pip install pybind11
pip install numpy==1.26.4

cd ../sddmm_module
./install_v64.sh


cd ../spmm_module
./install_v64.sh

cd ../performance_analysis
