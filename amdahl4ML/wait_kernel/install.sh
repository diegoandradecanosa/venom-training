pip uninstall -y wait_kernels
rm -rf build
rm -rf dist
rm -rf wait_kernels.egg-info
python3 -W ignore setup.py build
python3 -W ignore setup.py install