pip uninstall -y spatha_sddmm
rm -rf build
rm -rf dist
rm -rf spatha_sddmm.egg-info
python3 -W ignore setup_v64.py build
python3 -W ignore setup_v64.py install