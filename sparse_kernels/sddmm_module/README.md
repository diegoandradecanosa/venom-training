# SDDMM module

This folder contains the source files for the python module providing the SDDMM calls.

## Requirements

This module should be installed from a working CUDA environment, including a working installation of cuSPARSELt. The module was tested with CUDA 11.8 and cuSPARSELt 0.3.0.


## Installation

If you are using a V value of 64 in your VENOM formats, run the install_v64.sh script to uninstall an existing version of this module, compile and install this version in your python environment, as follows:

```
bash ./install_v64.sh
```

If, instead, you are using a V value of 128, you can install the version with no suffix:

```
bash ./install.sh
```

