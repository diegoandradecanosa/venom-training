# Amdahl4ML

## Goal and features.

This tool is designed to help visualize what parts of the model use more execution time, as well as create predictions of the effects of potential improvements to certain parts of the model, in order to be guide the efforts of optimization.

Using an ONNX representation of the model and, optional but recommended, the trace data of a forward pass of the model with profiling enabled, this tool generates a JSON file with relative execution time information (ratio of the total model time) as well as color data for loading into model explorer. Optionally, it can algo generate its own graph representation of the model, grouping the ONNX nodes into CUDA kernels as possible, using a library of operations describing the kernels that can be detected and the graph nodes they perform and the relation of those graph nodes. If the trace data is not provided, only the graph generation features are useful, since there is no time information for the rest of the features.

For the supported CUDA-backed operations, each node in the SVG output includes execution time, ratio of total execution time, as well as max theoretical speedup for the entire model is that node's execution time is instantaneous. This speedup is obviously not achievable, but it is provided to help developers focus their optimization efforts into the parts of the model that can provide greater return for the time invested. 

Where the model has multiple parallel paths, the SVG output of the tool includes the aggregated ratio of the total execution time for each branch, that is, how much of the total execution time all of the operations in that branch represent. The critical path of the model is also marked, showing the operations that determine the total execution time, asuming that parallel paths can be executed in parallel.

The tool can also generate two more SVG graphs, if desired. The first one is the full ONNX graph in SVG format, including data tensor nodes for the input and outputs of each operation. The second one is a simplified graph, removing those data tensor nodes and some other nodes that have no computation associated. This second one is the base for grouping operations for the CUDA kernels, this option is provided to be able to expand the library of operations supported.

The last major capability is the option to apply speedups to certain CUDA-backed operations, creating an updated graph representing how the time ratios, potential max theoretical speedups and critical path would change. This allows the analysis of what effect would a certain optimization would have on the model. This speedups can be provided to specific operations by name, or to all operations of the same type. If this option is enabled, on top of the updated model graph and JSON for model explorer, the tool also provides an estimated total execution time in text on the standard output.

Below is an example SVG output for a BERT mini layer si it can be easily read. The color indicates the ratio of total time each node represents, greener color indicate more costly operations where optimization efforts should be focused. The critical path is indicated as a dark red highligh of nodes and edges:


[//]: # ![Bert mini layer graph with critical highlighted](https://github.com/diego-teijeiro/venom-training/blob/main/amdahl4ML/bert-mini_profiling/bert-mini_layer_cuda_grouped_critical_path.svg?raw=true)


![Bert mini layer graph with critical highlighted](amdahl4ML/bert-mini_profiling/bert-mini_layer_cuda_grouped_critical_path.svg)

## Dependencies

In addition to the requirements to run your model, for the script that creates the graph, the python modules required are `onnx`, `pydot`, `graph_tool`, `model_explorer` and `colorsys`. 


## How to use

The first step is to create an ONNX representation of the model and profile an execution of the model in order to get the required timing information.

Exporting the model in an ONNX representation varies for different libraries and frameworks. For pytorch 2, a model stored in a variable named `model`, using an input in a variable named `input` can be exported to a file named 'model.onnx' in the following way:

```
torch.onnx.export(model, input, 'model.onnx', verbose=True)
```

In order to get the timing information, an execution is required in profiling mode, and the trace information has to be exported to a file. For a pytorch model, this can be done as follows, exporting the trace information to a file named `trace_dense.json`:
```
from torch.profiler import profile, record_function, ProfilerActivity
...
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
    with record_function("model_inference"):
        output = model(input)
prof.export_chrome_trace("trace_dense.json")
...
```

The next step is to launch this tool to parse the ONNX representation of the model, and create a graph. The trace file can be provided using an optional argument:

```
python custom_graph_drawer.py --rankdir LR --traces trace_dense.json --input model.onnx --output output.json --svg svg_file.svg
```

Required arguments:
* --input model.onnx: ONNX representation of the model to make the graph of.
* --output output.json: The output JSON file for Model Explorer.

Optional arguments: 
* --traces trace_dense.json: If provided, timing information is added to the graph, calculating speedups and colorizing the nodes based on the potential for optimization in a red-green gradient, as well as identifying the critical path in this execution.
* --rankdir LR: The direction of the graph, "TB" for top-to-bottom or vertically spread graph, or "LR" for left-to-right or horizontally spread graph. Default is "TB".
* --embed_docstring: Embed docstring as javascript alert. Useful for SVG format.
* --svg svg_file.svg: Export graph in SVG format to the indicated file.
* --speedups speedups.json: JSON with achievable speedups for different ops to generate updated timing information.
* --fullgraph full_graph.svg: SVG output file that shows all of the ONNX operations without grouping for CUDA kernels.
* --simplegraph simple_graph.svg: SVG output file that shows only the ONNX operations where grouping will be performed (removing input/output and data tensor graph nodes) but before the grouping step is performed.
* --debug: If present, prints debug information during the execution.


It is recommended to open the graph with a web browser to be able to clic on the nodes and see additional information, execution time and max theoretical speedup, for example.


