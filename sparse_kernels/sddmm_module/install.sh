pip uninstall -y spatha_sddmm
rm -rf build
rm -rf dist
rm -rf spatha_sddmm.egg-info
python3 -W ignore setup.py build
python3 -W ignore setup.py install