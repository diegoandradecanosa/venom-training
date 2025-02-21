# venom-training

This repository hosts several tools developed under the Inno4scale project Enabling SParse training of LLMs on GPUs (ESPLAG).

## Amdhal4ML

Software tool to help guide optimization efforts by providing visual representation of the execution time of each part of the model and critical path information, as well as total execution time predictions and updated visualizations of the execution time if indicated speedups for indicated operations are applied.

## Sparseml integration

### Description

The sparseml folder contains a fork of the sparseml repository, at version 1.7, that integrates the VENOM kernels as a pruning modifier and as a new pruning mask in the existing GMPruningModifier to apply the VENOM masks to tensors but not altering the kernels used for the execution, only creating the masks to apply to the tensors.

### Installation.

First, refer to the environment installation instructions on the performance_analysis folder for a more detailed guide on how to set up a working python environment for this project.

With a working environment, install the sparseml library with the additional modifier in development mode as follows:

```
cd sparseml
python3 -m pip install -e "./[dev]"
```

### Usage

In order to be able to use the VENOM masks for other pruning modifiers, simpy use a V:N:M sintax on the mask_type parameter of the pruning modifiers. For example, to configure a GMPruningModifier to use the VENOM mask of 64:2:8 use the following configuration in a recipe:
```
...
- !GMPruningModifier
...
  mask_type: "64:2:8"
...
```

This configuration only creates tensors masks that comply with the VENOM format, it does not compress the tensor or use the VENOM kernels. This is provided to enable the gradual pruning while maintaining mask compatibility with the VENOM masks that will be used on the last phase of the pruning where VENOM kernels can be used.

In order to exploit the sparsity in the tensors, a new pruning modifier was created, VENOMKernelReplacementPruningModifier. An example recipe snippet that changes to VENOM kernels at the start of the secon epoch, using a 64:2:4 mask is as follows:

```
  - !VENOMKernelReplacementPruningModifier
    params:
      - re:bert.encoder.layer.*.attention.self.query.weight
      - re:bert.encoder.layer.*.attention.self.key.weight
      - re:bert.encoder.layer.*.attention.self.value.weight
      - re:bert.encoder.layer.*.attention.output.dense.weight
      - re:bert.encoder.layer.*.intermediate.dense.weight
      - re:bert.encoder.layer.*.output.dense.weight
    at_epoch: 2
    leave_enabled: True
    mask_type: "64:2:4"
```