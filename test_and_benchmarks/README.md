
# Tests and benchmarks

This folder contains several scripts for testing the correctness of the venom kernels, as well as scripts to run model training for one epoch while measuring time. Additional required files are included in this folder so those scripts work correctly.

This folder also contains some helpful script to process result data, or perform specific small tests.

## Results
Results obtained with these scripts, as well as graphics derived from them, are stored in this folder.

## Short tests

The first set of tests are the bench_end2end scripts. These execute the model in the forward and backward pass to check the venom kernels work correctly. Using bench_end2end_dense.py as an example, to run it use the follwoing command:

```
python bench_end2end_dense.py -v 64 -n 2 -m 8 -bs 16
```

Commandline arguments:
* -v 64: Value for the V part of the V:N:M format.
* -n 2: Value for the N part of the V:N:M format.
* -m 8: Value for the M part of the V:N:M format.
* -bs 16: Batch size to use for the model.

## Epoch training

The next set of script are created to benchmark the execution time of an entire epoch, using dense computation and venom, with a variant that does not perform matrix transposition to measure the impact of them even though the gradients are not correct.

These scripts use DDP parallelism and should be executed in a valid distributed environment. To run the scripts you can use th efollowing command, using the dense version as an example:

```
python epoch_training_dense -v 64 -n 2 -m 8 -bs 16
```

The command arguments are the same as the short tests scripts.

