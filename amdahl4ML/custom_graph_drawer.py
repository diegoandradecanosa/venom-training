# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# A library and utility for drawing ONNX nets. Most of this implementation has
# been borrowed from the caffe2 implementation
# https://github.com/pytorch/pytorch/blob/master/caffe2/python/net_drawer.py
#
# The script takes two required arguments:
#   -input: a path to a serialized ModelProto .pb file.
#   -output: a path to write a dot file representation of the graph
#
# Given this dot file representation, you can-for example-export this to svg
# with the graphviz `dot` utility, like so:
#
#   $ dot -Tsvg my_output.dot -o my_output.svg
from __future__ import annotations
import time

#start_time = time.time()

import argparse
import json
from collections import defaultdict
from typing import Any, Callable, Dict
import re

import pydot
import graph_tool as gt

from onnx import GraphProto, ModelProto, NodeProto
from model_explorer import node_data_builder as ndb
import colorsys

from kernelOperations import operationsLibrary


CRITICAL_PATH_TEXTCOLOR='#00a7ff' # Text color for JSON export for Model explorer.
CRITICAL_PATH_SVG_COLOR='#0000ff' # Color to mark critical path in SVG export image.

CUDA_OP_STYLE = {
    "shape": "box",
    "color": "#0F9D58",
    "style": "filled",
    "fontcolor": "#FFFFFF",
}
# Additional properties for nodes that will be added by the code:
#   op_type: The type of operation of the node.
#   op_id: The id of the operation, to provide some ordering.
#   node_path: The full name of the operation, including layer names separated by slashes (/). {Op name}/{op type}
#   extra_text: Text to show when clicked. UpdateURL sets the URL property that shows this text.
#   critical_path: Whether or not the node/edge is part of the critical path.
#   branch_aggregated_ratio: The ratio of total time the entire branch represents.
#   start_time: The timestamp of the moment the node starts execution.
#   duration: The duration, in microseconds (µs) of the node.
#   ratio_of_cuda_time: The ratio of the total time that this node represents. duration/total_time.
#   max_speedup: Maximum speedup achievable if this node has instant execution. Theoretical, only informative to prioritize optimizations.
#   nodes_path: paths of all ONNX nodes grouped under this node, comma-separated.
# 
# Additional properties for edges that will be added by the code:
#   branch_aggregated_ratio

CPU_OP_STYLE = {
    "shape": "hexagon",
    "color": "#f0f0f0",
    "style": "filled",
    "fontcolor": "#000000",
}

INPUT_STYLE = {"shape": "octagon",
    "op_type": "input",
    "op_id": -1,
}



# Operation types that create new inputs (no input on the operation but the output is used). 
# Required for cuda-only graph since these are not backed by kernels, but the graph breaks otherwise.
# These operations will not be shown in the graph.



_NodeProducer = Callable[[NodeProto, int], pydot.Node]

def convert_microseconds(microseconds):
        if microseconds >= 60_000_000:
            # Convert microseconds to minutes
            minutes = microseconds / 60_000_000.0
            return f"{minutes} minutes"
        elif microseconds >= 1_000_000:
            # Convert microseconds to seconds
            seconds = microseconds / 1_000_000.0
            return f"{seconds} seconds"
        elif microseconds >= 1_000:
            # Convert microseconds to milliseconds
            milliseconds = microseconds / 1000.0
            return f"{milliseconds} milliseconds"
        else:
            return f"{microseconds} microseconds"


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()

def _escape_label(name: str) -> str:
    # json.dumps is poor man's escaping
    return json.dumps(name)


def _form_and_sanitize_docstring(s: str) -> str:
    url = "javascript:alert("
    url += _escape_label(s).replace('"', "'").replace("<", "").replace(">", "")
    url += ")"
    return url


# reads provided json file and processes it, storing the traces of the cuda kernels in a list that is returned.
def loadJSONCudaTraces(
    json_file: str # Json file to load
    ): 

    # Load JSON data from a file
    with open(json_file) as f:
        data = json.load(f)
    
    # Use list comprehension to filter trace events to ones that have pid 0 and "kernel" in cat attribute.    
    cuda_traces = [event for event in data['traceEvents'] if event.get('pid') == 0 and event.get('cat') == "kernel" ]

    return cuda_traces

def createGraphToolGraph(name:str = None) -> gt.Graph:
    """Creates a new graph_tool graph, with the required graph, vertex and edge property maps already set as internal properties.
    Graph properties included:
        name (string): The name for the graph.
        max_ratio_of_cuda_time (double): The highest value for the vertex property ratio_of_cuda_time
    Vertex properties included:
        label (string): The text to use as label for the vertex.
        color (string): The color to use for the vertex. 
        fontcolor (string): The color to use for the text on the vertex. 
        shape (string): The pydot shape name to use for the vertex. 
        style (string): The style property to apply on the vertex. 
        penwidth (int32_t): Thickness for drawing lines for this vertex.
        fillcolor (string): The color to use for the fill the shape of the vertex. 
        op_type (string):  The type of operation of the node.
        op_id (int32_t): The id of the operation, to provide some ordering.
        name (string): The name of the vertex. Combination of op_type and op_id.
        parallel_paths (int): The amount of paths in the graph, other that though this vertex, that can be used to reach later parts of the graph. 0 means every paths goes though this vertex.
        node_path (string): The full name of the operation, including layer names separated by slashes (/). {Op name}/{op type}
        extra_text (string): Text to show when clicked. UpdateURL sets the URL property that shows this text.
        critical_path (bool): Whether or not the node/edge is part of the critical path.
#        branch_aggregated_ratio (double): The ratio of total time the entire branch represents.
        start_time (int64_t): The timestamp of the moment the node starts execution.
        duration (int32_t): The duration, in microseconds (µs) of the node.
        ratio_of_cuda_time (double): The ratio of the total time that this node represents. duration/total_time.
        max_speedup (double): Maximum speedup achievable if this node has instant execution. Theoretical, only informative to prioritize optimizations.
        nodes_path (string): paths of all ONNX nodes grouped under this node, comma-separated.
    Edge  properties included:
        label (string): The label to put next to the edge. 
        color (string): The color to use for the edge. 
        style (string): The style property to apply on the edge. 
        branch_aggregated_ratio (double): The ratio of total time the entire branch represents.
        critical_path (bool): Whether or not the node/edge is part of the critical path.

    Returns:
        gt.Graph: _description_
    """
    graph = gt.Graph()
    graph_prop = graph.new_graph_property('string')
    graph.gp['name'] = graph_prop # Add property to graph
    if name is not None:
        graph.gp['name'] = name # Set value to name property.

    graph_prop = graph.new_graph_property('double')
    graph.gp['max_ratio_of_cuda_time'] = graph_prop
    # Add vertex properties to the graph
    vertex_prop = graph.new_vertex_property('string')
    graph.vp['label'] = vertex_prop
    vertex_prop = graph.new_vertex_property('string', 'black')
    graph.vp['color'] = vertex_prop
    vertex_prop = graph.new_vertex_property('string', '#FFFFFF')
    graph.vp['fontcolor'] = vertex_prop
    vertex_prop = graph.new_vertex_property('string', 'box')
    graph.vp['shape'] = vertex_prop
    vertex_prop = graph.new_vertex_property('string')
    graph.vp['style'] = vertex_prop
    vertex_prop = graph.new_vertex_property('int32_t')
    graph.vp['penwidth'] = vertex_prop
    vertex_prop = graph.new_vertex_property('string')
    graph.vp['fillcolor'] = vertex_prop
    vertex_prop = graph.new_vertex_property('string')
    graph.vp['op_type'] = vertex_prop
    vertex_prop = graph.new_vertex_property('int32_t', -1)
    graph.vp['op_id'] = vertex_prop
    vertex_prop = graph.new_vertex_property('string')
    graph.vp['name'] = vertex_prop
    vertex_prop = graph.new_vertex_property('int32_t')
    graph.vp['parallel_paths'] = vertex_prop
    vertex_prop = graph.new_vertex_property('string')
    graph.vp['node_path'] = vertex_prop
    vertex_prop = graph.new_vertex_property('string')
    graph.vp['extra_text'] = vertex_prop
    vertex_prop = graph.new_vertex_property('bool')
    graph.vp['critical_path'] = vertex_prop
    vertex_prop = graph.new_vertex_property('int64_t')
    graph.vp['start_time'] = vertex_prop
    vertex_prop = graph.new_vertex_property('int32_t')
    graph.vp['duration'] = vertex_prop
    vertex_prop = graph.new_vertex_property('double')
    graph.vp['ratio_of_cuda_time'] = vertex_prop
    vertex_prop = graph.new_vertex_property('double')
    graph.vp['max_speedup'] = vertex_prop
    vertex_prop = graph.new_vertex_property('python::object')
    graph.vp['nodes_path'] = vertex_prop
    # Add edge property maps
    egde_prop = graph.new_edge_property('string')
    graph.ep['label'] = egde_prop
    egde_prop = graph.new_edge_property('string')
    graph.ep['color'] = egde_prop
    egde_prop = graph.new_edge_property('string')
    graph.ep['style'] = egde_prop
    egde_prop = graph.new_edge_property('double', val=-0.1)
    graph.ep['branch_aggregated_ratio'] = egde_prop
    egde_prop = graph.new_edge_property('bool')
    graph.ep['critical_path'] = egde_prop

    return graph


def addVertexToGraph(graph:gt.Graph, **kwargs) -> gt.Vertex:

    # Create new vertex in the graph
    new_vertex = graph.add_vertex()

    # Set properties for the new vertex using kwargs
    for key, value in kwargs.items():
        graph.vp[key][new_vertex] = value

        # Set vertex name if kwargs contains both op_type and op_id
    if 'op_type' in kwargs.keys() and 'op_id' in kwargs.keys() and ('name' not in kwargs.keys() or kwargs['name'] is None or kwargs['name'] == '') :
        # Name is not provided but op_type and id are. Generate name for vertex.
        op_type = kwargs['op_type']
        op_id = kwargs['op_id']
        graph.vp['name'][new_vertex] = f"{op_type} (op#{op_id})"

    return new_vertex


def get_vertex_properties(graph:gt.Graph, vertex:gt.Vertex):
    """Returns all of the properties of the given vertex in the graph.

    Args:
        graph (gt.Graph): The graph containing the vertex.
        vertex (gt.Vertex): The vertex to get the properties of.

    Returns:
        map: A map of property names and their value for the given vertex, as stored in the graph's vertex propoerties.
    """
    properties = {}
    for prop_name, prop_map in graph.vertex_properties.items():
        properties[prop_name] = prop_map[vertex]

    return properties

def get_edge_properties(graph:gt.Graph, edge:gt.Edge):
    """Returns all of the properties of the given edge in the graph.

    Args:
        graph (gt.Graph): The graph containing the vertex.
        edge (gt.Edge): The edge to get the properties of.

    Returns:
        map: A map of property names and their value for the given edge, as stored in the graph's edge propoerties.
    """
    properties = {}
    for prop_name, prop_map in graph.edge_properties.items():
        properties[prop_name] = prop_map[edge]
    return properties

def GetOpNodeProducer(  # noqa: N802
    embed_docstring: bool = False, **kwargs: Any
) -> _NodeProducer:
    def really_get_op_node(op: NodeProto, op_id: int, op_duration: int) -> pydot.Node:
        node_name = f"{op.op_type} (op#{op_id})"
        
        
        node = pydot.Node(node_name, **kwargs)
        url = "Inputs:"
        for i, input_ in enumerate(op.input):
            url += "\n\tinput" + str(i) + " " + input_
        url += "\nOutputs: "
        for i, output in enumerate(op.output):
            url += "\n\toutput" + str(i) + " " + output
	    
        node.set('extra_text', url)
        node.set_URL(_form_and_sanitize_docstring(url))
        node.set("op_type", op.op_type)
        node.set('op_id', op_id)
        node.set('node_path', op.name)
        return node

    return really_get_op_node

def GetGroupedNodeProducer(  # noqa: N802
    embed_docstring: bool = False, **kwargs: Any
) -> _NodeProducer:
    def really_get_op_node(op: str, op_id: int, nodes_path: list) -> pydot.Node:
        node_name = f"{op} (op#{op_id})"
        
        # Construct new url information, inputs and outputs. Inputs from first node in the group, outputs from last in group.
        node = pydot.Node(node_name, **kwargs)
        
        node.set('op_type', op)
        node.set('op_id', op_id)
        node.set('nodes_path', nodes_path)
        return node

    return really_get_op_node

# Creates a new pydot.Node with the same op_type and other data but a new id.
def GetOpNodeDuplicator(  # noqa: N802
    embed_docstring: bool = False, **kwargs: Any
) -> _NodeProducer:
    def really_get_op_node(source: pydot.Node, op_id: int) -> pydot.Node:
        # Recreate node name with new ID
        op_type = source.get('op_type')
        if op_type=='input':
            # Input node, copy full name since it does not have id attached.
            node_name=source.get_name()
            args = INPUT_STYLE
        else:
            # Normal operation node, rebuild new name with new id.
            node_name = f"{op_type} (op#{op_id})"
            args = kwargs
        
        # Create node and copy extra_text attribute, setting it as URL dialog content.
        node = pydot.Node(node_name, **args)
        # Copy all attributes
        for attribute, value in source.get_attributes().items():
            node.set(attribute, value)   
        # Overwrite some attributes.
        node.set('op_type', op_type)
        if op_type=='input':
            node.set('op_id', -1)
        else:
            node.set('op_id', op_id)

        return node

    return really_get_op_node

# Duplicates a node only storing information to export to svg. Name (label), color, fontcolor, shape, style and extra_text as URL.
def GetOpNodeSanitizer(  # noqa: N802
) -> _NodeProducer:
    def really_get_op_node(source: pydot.Node) -> pydot.Node:
        node = pydot.Node(source.get_name())
        source_attributes_keys = source.get_attributes().keys()
        if 'color' in source_attributes_keys:
            node.set('color', source.get('color'))
        if 'fontcolor' in source_attributes_keys:
            node.set('fontcolor', source.get('fontcolor'))
        if 'shape' in source_attributes_keys:
            node.set('shape', source.get('shape'))
        if 'style' in source_attributes_keys:
            node.set('style', source.get('style'))
        if 'extra_text' in source_attributes_keys:
            node.set('extra_text', source.get('extra_text'))
        if 'duration' in source_attributes_keys:
            node.set('duration', source.get('duration'))
        if 'max_speedup' in source_attributes_keys:
            node.set('max_speedup', source.get('max_speedup'))
        if 'penwidth' in source_attributes_keys:
            node.set('penwidth', source.get('penwidth'))
        if 'fillcolor' in source_attributes_keys:
            node.set('fillcolor', source.get('fillcolor'))

        updatePydotNodeURL(node)
        return node
    return really_get_op_node

# Duplicates an edge only storing information to export to svg. label, color, and style.
def GetEdgeSanitizer(  # noqa: N802
) -> _NodeProducer:
    def really_get_edge(source: pydot.Edge) -> pydot.Edge:
        edge = pydot.Edge(source.get_source(), source.get_destination())
        source_attributes_keys = source.get_attributes().keys()
        if 'label' in source_attributes_keys and source.get('label') != '':
            edge.set('label', source.get('label'))
        if 'color' in source_attributes_keys and source.get('color') != '':
            edge.set('color', source.get('color'))
        if 'style' in source_attributes_keys and source.get('style') != '':
            edge.set('style', source.get('style'))
        return edge
    return really_get_edge

# Updates the URL of the node. When the node is clicked, a new window opens to show extra information. 
# If the node contains extra text, it is included first. Input and outputs are expected here.
# If duration is available for the node, it is included next.
# If max speedup data is available, it is added to the extra information given.
def updatePydotNodeURL(
    node: pydot.Node,
):

    node_attributes = node.get_attributes()
    url = node.get('extra_text') if 'extra_text' in node_attributes.keys() else ''
    # Add duration if available.
    if 'op_id' in node_attributes.keys() and node.get('op_id') is not None:
        url += "\nop_id: "+ str(node.get('op_id'))
    # Add duration if available.
    if 'duration' in node_attributes.keys() and node.get('duration') is not None:
        url += "\nduration: "+ str(node.get('duration'))+ " µs"
    # Add max speedup if available
    if 'max_speedup' in node_attributes.keys() and node.get('max_speedup') is not None:
        url += "\nMax theoritical speedup: " + "{:.3f}".format(node.get('max_speedup'))
    # Add wheter or not the node is part of the critical path
    
    # Only set url if it has content.
    if url is not None and len(url) > 0:
        node.set('URL', _form_and_sanitize_docstring(url))

# Checks if node has properties that cause the render style to change.
# If node is in critical path, style is set to bold to indicate the node is part of the critical path.
# NOT SURE IF NECESSARY: If the node is and alternative critical path (critical path is another parallel path), use dotted lines to differentiate. style=dotted
#       All nodes in critical path use thicker lines, the rest of the nodes are normal-thickness and would all be dotted.
# If node is not on critical path, change style to striped to indicate not interesting target of optimization but still contain optimization potential.
# Change color based on ratio of the total time that this node represents, with respect to the single slowest node.
def updatePydotNodeStyle(
    node: pydot.Node,
    slowest_node_ratio_of_total_time: float,
):
    node_attributes = node.get_attributes()
    node.set('color', '#000000') # Set line color to black.
    # If part of critical path, set style to bold and set color according to optimization potential
    if 'critical_path' in node_attributes.keys() and node.get('critical_path'):
        node.set('penwidth', 3)
        node.set('color', CRITICAL_PATH_SVG_COLOR)
        if slowest_node_ratio_of_total_time != 0 and 'ratio_of_cuda_time' in node_attributes.keys() and node.get('ratio_of_cuda_time') is not None:
            # Node has ratio of total cuda time. Adjust color 'color': '#f0f0f0',
            node.set('fillcolor', value_to_color_green_to_red(node.get('ratio_of_cuda_time'), slowest_node_ratio_of_total_time))
        else:
            # Set a blue color to indicate no timing information available.
            node.set('fillcolor', '#34ebeb')
            node.set('fontcolor', 'black')
    elif 'op_type' in node_attributes.keys() and node.get('op_type') != 'input':
        # Node not in critical path. Set style to striped and colorlists to optimization potential and grey.
        node.set('style', 'striped')
        if slowest_node_ratio_of_total_time != 0:
            optimization_color = value_to_color_green_to_red(node.get('ratio_of_cuda_time'), slowest_node_ratio_of_total_time)
            node.set('fillcolor', optimization_color+':#d0d0d0:'+optimization_color+':#d0d0d0:'+optimization_color+':#d0d0d0')
        else:
            # No timing information given, use blue color
            node.set('fillcolor', '#34ebeb')
            node.set('fontcolor', 'black')
    else:
        # it is an input node. Set line color and style filled, set label to name.
        node.set('label', node.get('name')) 
        node.set('label', 'octagon')
        node.set('color', 'black')
        node.set('penwidth', 1)



# Checks if the edge has properties that cause the render style to change.
def updatePydotEdgeStyle(
    edge: pydot.Edge,
):
    edge_attributes = edge.get_attributes()
    if 'critical_path' in edge_attributes.keys() and edge.get('critical_path'):
        edge.set('style', 'bold')
        edge.set('color', CRITICAL_PATH_SVG_COLOR)
    if 'branch_aggregated_ratio' in edge_attributes.keys() and edge.get('branch_aggregated_ratio') is not None and edge.get('branch_aggregated_ratio') >= 0:
        edge.set('label', "{:.2f}".format(edge.get('branch_aggregated_ratio')))

def value_to_color_green_to_red(value, max_value):
    """
    Transforms a value between 0 and max_value to a color from green to red.
    
    :param value: The input value, should be between 0 and max_value.
    :param max_value: The maximum value in the range.
    :return: A string representing the color in hex format.
    """
    if not (0 <= value <= max_value):
        raise ValueError("Value must be between 0 and max_value")
    
    # Calculate the interpolation factor (0.0 to 1.0)
    ratio = value / max_value
    
    # Interpolate between green (0, 255, 0) and red (255, 0, 0)
    red = int(255 * ratio)
    green = int(255 * (1 - ratio))
    blue = 0
    
    # Convert the RGB values to a hex color code
    color = f"#{red:02x}{green:02x}{blue:02x}"
    
    return color


# Get a color in a red->green gradient from a number and the max value for the number.
def value_to_color_hex_red_to_green(value, max_value):
    # Normalize the value to the range [0, 1]
    normalized_value = value / max_value
    
    # Map the normalized value to the hue range [0, 120]
    hue = normalized_value * 120
    
    # Convert HSV color to RGB
    rgb = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
    
    # Scale RGB values to the range [0, 255] and convert to integers
    r = int(rgb[0] * 255)
    g = int(rgb[1] * 255)
    b = int(rgb[2] * 255)
    
    # Convert RGB values to hexadecimal format
    color_hex = "#{:02x}{:02x}{:02x}".format(r, g, b)
    
    # Return the hexadecimal color string
    return color_hex




def identifyCriticalPath(
    graph: gt.Graph,
    debug: bool = False,
):

    # Locate the node where the edges provided meet going back to the start of the graph.
    # It uses the nodes ID to guide edge exploration. 
    def locate_bifurcating_node(
        graph: gt.Graph,
        joining_vertex: gt.Vertex, 
        debug: bool = False,
    ) -> pydot.Node:

        # Keep track of the lowest node ID found. Only branches that have not reached this ID are explored. 
        # Get the lowest id from the joining edges.
        lowest_id = None
        for vertex in joining_vertex.in_neighbors():
            if lowest_id is None or graph.vp['op_id'][vertex] < lowest_id:
                lowest_id = graph.vp['op_id'][vertex]
                
        print('\t\tLowest ID from the in-neighbors:', lowest_id) if debug else None

        # Store the nodes of explored edges sources. When this list has only one, the search can stop.
        vertices_to_explore = list(joining_vertex.in_neighbors())

        while len(vertices_to_explore) > 1:
            print('\t\tVertices to explore:', [ (graph.vp['name'][vertex], graph.vp['op_id'][vertex]) for vertex in vertices_to_explore]) if debug else None
            # Iterate over the nodes to explore, if the ID is higher than lowest_id, locate edge(s) leading to it, add edge sources to nodes to explore.
            new_vertices = []
            explored_vertices = []
            for vertex in vertices_to_explore:
                if graph.vp['op_id'][vertex] > lowest_id:
                    print('\t\tVertex is above lowest id(', lowest_id, '), checking edges leading to', graph.vp['name'][vertex]) if debug else None
                    
                    for neighbor in vertex.in_neighbors():
                        if neighbor not in new_vertices:
                            print('\t\t\tNew vertex to explore: ', graph.vp['name'][neighbor]) if debug else None
                            new_vertices.append(neighbor)
                            # Check if edge source has a lower id than current lowest
                            if graph.vp['op_id'][neighbor] < lowest_id:
                                print('\t\t\tLowest ID decreasing from', lowest_id, 'to', graph.vp['op_id'][neighbor]) if debug else None
                                lowest_id = graph.vp['op_id'][neighbor]
                    
                    explored_vertices.append(vertex)
            # For end. Remove explored nodes from nodes_to_explore and include new nodes. Done here to avoid changing the list while in for loop.
            for explored_node in explored_vertices:
                vertices_to_explore.remove(explored_node)
            added = 0
            for new_node in new_vertices:
                if new_node not in vertices_to_explore:
                    added+=1
                    vertices_to_explore.append(new_node)
            print('\t\tRemoved', len(explored_vertices), 'nodes from the nodes to explore, added', added,' new ones.') if debug else None
            if len(explored_vertices) == 0 and added == 0:
                # Not nodes added or removed, stuck.
                print('\t\tNo new or removed nodes, loop seems stuck, throtling output...') if debug else None
                time.sleep(1)
        print('\t\tOnly 1 node remains in nodes_to-explore: ', vertices_to_explore[0]) if debug else None
        return vertices_to_explore[0]

    # Traverses the graph and aggregates the ratios of total time of the nodes from path_start to path_end nodes arriving at path_end through the provided edge.
    def aggregate_ratio_of_total_time(
        last_edge: gt.Edge,
        graph: gt.Graph,
        path_start: gt.Vertex,
        path_end: gt.Vertex,
        debug: bool = False,
    )-> (float, int):
        """_summary_

        Args:
            gtlast_edge (gt.Edge): _description_
            gtgraph (gt.Graph): _description_
            gtpath_start (gt.Vertex): _description_
            gtpath_end (gt.Vertex): _description_
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            float, int: branch_aggregated_ratio, branch_size (vertex amount in the branch)
        """
        # Node-less branch, this branch goes from start to end directly.
        if last_edge.source() == path_start:
            print('\t\tVertex-less branch, 0 time and 0 nodes counted') if debug else None
            return 0, 0

        aggregated_ratio_of_total_time = 0

        # Keep track of the next node to aggregate
        next_vertex_in_branch =last_edge.source()
        counted_nodes = 1

        print('\t\tAggregating ratio for branch:', graph.vp['name'][path_start], 'to', graph.vp['name'][path_end], 'via', graph.vp['name'][last_edge.source()]) if debug else None
        while next_vertex_in_branch != path_start:
            # Aggregate next node.
            aggregated_ratio_of_total_time += graph.vp['ratio_of_cuda_time'][next_vertex_in_branch]
            # Check edges that point to next_node_in_branch.
            next_edges =  next_vertex_in_branch.in_edges()
            if next_vertex_in_branch.in_degree() > 1:
                print('\t\tFound sub-branch, locating branch start and using ratio of the slowest...') if debug else None
                # Several branches converge on this node. The aggregated ratio of the branch will use the largest ratio of the branched that end here. Recursive call.
                branch_start = locate_bifurcating_node(graph, next_vertex_in_branch, debug=debug)
                # Calculate the aggregated ratio of total time of each branch.
                slowest_branch_ratio = None
                #slowest_branch_edge = None
                for next_edge in next_edges:
                    branch_aggregated_ratio, nodes_in_branch = aggregate_ratio_of_total_time(next_edge, graph, branch_start, next_vertex_in_branch, debug=debug)
                    counted_nodes += nodes_in_branch
                    if slowest_branch_ratio is None or branch_aggregated_ratio > slowest_branch_ratio:
                        slowest_branch_ratio = branch_aggregated_ratio
                        #slowest_branch_edge = edge
                # Add ratio form slowest branch to current aggregation
                aggregated_ratio_of_total_time += slowest_branch_ratio
                # Next in branch is the start of the subbranches.
                next_vertex_in_branch = branch_start
            elif next_vertex_in_branch.in_degree() == 1:
                # keep going back on the branch.
                next_vertex_in_branch = next(next_edges).source()
                # Increase counted nodes if it is not the start of the branch.
                if next_vertex_in_branch != path_start:
                    counted_nodes += 1
            else:
                # No more edges, error.
                print('No more edges to follow while aggregating ratios of total time for critical path identification.')
                exit()

        print('\t\tThis branch represents', aggregated_ratio_of_total_time*100, 'percent of the total cuda time and', counted_nodes,'nodes.') if debug else None
        return aggregated_ratio_of_total_time, counted_nodes





    # Get the list of nodes and edges in the graph for repeated use
    #graph_vertices = graph.vertices()

    # Identify first and last nodes. Nodes with no edges pointing to them and from them, respectively.
    for vertex in graph.vertices():
        # Delete potetnial previous critical path information
        graph.vp['critical_path'][vertex]=False
        if vertex.out_degree() == 0:
            # This node has no edges coming out of it, it is the last node.
            last_vertex = vertex
        if vertex.in_degree() == 0:
            # This node has no edges ending in it, it is the first node of the graph.
            first_vertex = vertex
    # For end. Check that we have last and first nodes.
    if last_vertex is None or first_vertex is None:
        print('Could not identify first and last node of the graph using the edges. Critical path identification not possible.')
        return
    
    for edge in graph.edges():
        graph.ep['critical_path'][edge]=False
        graph.ep['branch_aggregated_ratio'][edge]=-1 # Negative value to hide labels in all edges, will be replaced for the relevant edges.

    # Start critical path identification. Last node is always in critical path
    remaining_vertices_amount = graph.num_vertices()
    graph.vp['critical_path'][last_vertex] = True
    remaining_vertices_amount-=1
    critical_path_front = last_vertex
    while remaining_vertices_amount>0:
        # Find edge(s) pointing to current critical path front
        print('\tCurrent critical path front: ', graph.vp['name'][critical_path_front]) if debug else None
        print('\t', critical_path_front.in_degree(), ' edges pointing to current critical path front.', sep='') if debug else None
        if critical_path_front.in_degree()==1:
            # If only one edge is found, simple case, mark edge and edge source as critical path, move critical path front to egde source.
            edge = next(critical_path_front.in_edges())
            graph.ep['critical_path'][edge] = True
            critical_path_front = next(critical_path_front.in_neighbors())
            graph.vp['critical_path'][critical_path_front] = True
            remaining_vertices_amount-=1
        elif critical_path_front.in_degree()>1:
            # Multiple edges point to current critical path front. Bifurcation, one of th edges will be critical path, the slowest one
            print('\tBifurcation in critical path found.', remaining_vertices_amount, 'remaining nodes to check.') if debug else None
            # Mark node where the edges combine as critical path, reduce the remaning node count by 1
            remaining_vertices_amount-=1 
            # Locate the node where the branches started.
            rejoining_node = locate_bifurcating_node(graph, critical_path_front, debug=debug)
            print('\tBifurcated paths converge on node', graph.vp['name'][rejoining_node]) if debug else None
            # Calculate the aggregated ratio of total time of each branch.
            slowest_branch_ratio = None
            slowest_branch_edge = None
            slowest_branch_node_count = 0
            nodes_in_branches = 0
            for edge in critical_path_front.in_edges():
                branch_aggregated_ratio, nodes_in_branch = aggregate_ratio_of_total_time(edge, graph, rejoining_node, critical_path_front, debug=debug)
                nodes_in_branches += nodes_in_branch
                # Label edge with their aggregated ratio of total time.
                graph.ep['branch_aggregated_ratio'][edge] = branch_aggregated_ratio
                if slowest_branch_ratio is None or branch_aggregated_ratio > slowest_branch_ratio:
                    slowest_branch_ratio = branch_aggregated_ratio
                    slowest_branch_edge = edge
                    slowest_branch_node_count = nodes_in_branch
            print('\tCritical path has', slowest_branch_node_count, 'nodes, while faster branch(es) have', (nodes_in_branches - slowest_branch_node_count), 
                    'nodes. Critical path continues on node', graph.vp['name'][slowest_branch_edge.source()]) if debug else None
            # Update remaining_node_amount. All of the nodes on faster branches, but leave the amount of nodes in the critical one to keep traveling and marking them as critical.
            remaining_vertices_amount -= (nodes_in_branches - slowest_branch_node_count)
            print('\t\t'+str(nodes_in_branches - slowest_branch_node_count), 'nodes processed,', remaining_vertices_amount, 'pending to check.') if debug else None
            # For the slowest branch, mark edge as critical and set edge source as critical_path_front.
            graph.ep['critical_path'][slowest_branch_edge] = True
            critical_path_front = slowest_branch_edge.source()
            graph.vp['critical_path'][critical_path_front] = True
            

            #remaining_node_amount = 0
        else:
            # Node with no edges pointing to it, assumed the first node in the graph, end of critical path but reminaing nodes is not 1 (input)?
            if remaining_vertices_amount>1:
                print('Reached node with no edges pointing to it (first in the graph) but reamining nodes to check is greater than 1:', remaining_vertices_amount)
                exit()
    # While end


# Creates the JSON to load the timing/speedup information to Model explorer as custom node data. 
# Currently exports max theoretical speedup and sets background color accordingly, critical path is exported as text color. 
# TODO: Investigate if multiple custom data is possible in the same JSON, to improve critical path visualization in model explorer.
def exportTimingDataToJSON(
    graph: gt.Graph,
    onnx_graph: GraphProto,
    output_json_file: str,
    debug: bool = False,
):
    # Initialize JSON structure
    results: Dict[str, ndb.NodeDataResult] = {}

    # Get the largest ratio of total time
    largest_ratio_of_total_time = 0
    for vertex in graph.vertices():
        if graph.vp['ratio_of_cuda_time'][vertex] > largest_ratio_of_total_time:
            largest_ratio_of_total_time = graph.vp['ratio_of_cuda_time'][vertex]
    
    # Iterate over nodes in the graph, writing the data for each node.
    for vertex in graph.vertices():
        print('Processing vertex', graph.vp['name'][vertex]) if debug else None
        if graph.vp['ratio_of_cuda_time'][vertex] != 0:
            # Set ratio of total time
            vertex_ratio = graph.vp['ratio_of_cuda_time'][vertex]
            bgColor = value_to_color_green_to_red(vertex_ratio, largest_ratio_of_total_time)
            # Add to GraphNodeData
            print('\tSetting nodeResult for vertex', graph.vp['node_path'][vertex], ' to value', vertex_ratio, 
                    'and color', value_to_color_green_to_red(vertex_ratio, largest_ratio_of_total_time)) if debug else None
            if graph.vp['node_path'][vertex] != ""  and graph.vp['critical_path'][vertex]:
                # Non grouped node. Add new nodeResult with ratio color and, if in critical path, text color.
                nodeResult = ndb.NodeDataResult(value=vertex_ratio, bgColor=bgColor)
                print('\t\tAdding vertex to results:', graph.vp['node_path'][vertex], '->', str(nodeResult)) if debug else None
                results[graph.vp['node_path'][vertex]] = nodeResult
            # Check if node is grouped operation and has paths for several nodes.
            if graph.vp['nodes_path'][vertex] is not None and graph.vp['critical_path'][vertex]:
                # Grouped node. Add several nodes with the same data form the group
                for path in graph.vp['nodes_path'][vertex]:
                    nodeResult = ndb.NodeDataResult(value=vertex_ratio, bgColor=bgColor)
                    print('\t\tAdding grouped vertex data to node path:', path, '->', str(nodeResult)) if debug else None
                    results[path] = nodeResult

    node_data = ndb.GraphNodeData(results)

    # Write JSON file.
    models = ndb.ModelNodeData({onnx_graph.name: node_data})
    models.save_to_file(output_json_file)

    
def calculate_cuda_wall_time(
    graph: gt.Graph,
    debug: bool = False,
):

    def combine_overlapping_time_slots(
        time_slots: Dict,
        debug: bool = False,
    ) -> bool :
        """
        Goes over the time slots, and combines the first one that overlaps another one. 
        Only combines time slots once and returns inmediatly to avoid iterating over collections being modified.
        Returns True if it has combined any slot, False is there is no more overlapping slots to combine.
        """
        # Check all slots
        for slot_start_time, slot_duration in time_slots.items():
            # Check against all time slots with different startime
            for other_slot_start_time, other_slot_duration in time_slots.items():
                # Check if slot starts during another time slot, overlapping slots, combine them and return.
                if slot_start_time != other_slot_start_time and slot_start_time > other_slot_start_time and slot_start_time <= other_slot_start_time+other_slot_duration:
                    # Check if first slot end time is later that other slot end time. 
                    if (slot_start_time + slot_duration ) > (other_slot_start_time+other_slot_duration):
                        # Use this first slot endtime for the other slot so it encompases both time slots.
                        new_duration = slot_start_time + slot_duration - other_slot_start_time # End timestamp minus start timestamp.
                        print('\t\t\tExpanding time slot with start', other_slot_start_time, 'from duration', time_slots[other_slot_start_time], 'to duration', new_duration) if debug else None
                        # Replace duration of other time slot.
                        time_slots[other_slot_start_time] = new_duration
                    # Else: First slot end before the other (bigger) slot, it is completely covered by the other slot.
                    # At this point, the other slot covers the first one, either it did from the start (else branch) or the other slot was expanded to cover the first one (if branch). Remove this slot form the list and return True
                    print('\t\t\tRemoving time slot starting at', slot_start_time, ' with duration', slot_duration, 'due to being part of slot starting at', other_slot_start_time, 'with duration', time_slots[other_slot_start_time]) if debug else None
                    time_slots.pop(slot_start_time)
                    return True

                        
        # No overlapping slots found, return False.
        return False

    
    """
    Uses the start time and duration of each node in the graph to calculate the precise wall time of executing all nodes in the graph, taking into account
    """

    busy_time_slots = dict()
    # Add a time slot for each node with start time and duration, them combine them until no more can be combined.
    for vertex in graph.vertices():
        # Only process nodes with start time and duration  
        if graph.vp['start_time'][vertex] is not None and graph.vp['duration'][vertex]:
            # Put all time slots in.
            busy_time_slots[graph.vp['start_time'][vertex]] = graph.vp['duration'][vertex]
            print('\t\tCreating new time slot with start', graph.vp['start_time'][vertex], 'and duration', graph.vp['duration'][vertex]) if debug else None
    # While overlaping time slots exists, combine them.
    combined_slots = 0
    while combine_overlapping_time_slots(busy_time_slots, debug=debug):
        # Count how many slots are combined to have something to do inside the loop.abs
        combined_slots += 1
    print('\t\tCombined time slots:', combined_slots) if combined_slots > 0 and debug else None

    # Aggregate the durations of all the time slots.
    cuda_busy_time = 0
    first_slot_startime = None
    last_slot_startime = 0
    last_slot_duration = None
    for slot_start_time, slot_duration in busy_time_slots.items():
        cuda_busy_time += slot_duration
        if first_slot_startime is None or slot_start_time < first_slot_startime:
            first_slot_startime = slot_start_time
        if slot_start_time > last_slot_startime:
            last_slot_startime = slot_start_time
            last_slot_duration = slot_duration
    
    cuda_wall_time = (last_slot_startime+last_slot_duration)-first_slot_startime

    return cuda_busy_time, cuda_wall_time

def updateTimeRatiosOfTotal(
    graph: gt.Graph,
    operations_library: operationsLibrary,
    total_cuda_time,
)-> float:
    """_summary_

    Args:
        graph (pydot.Dot): The graph to update the ratios of.
        operations_library (operationsLibrary): The possible operations on the graph. Used to extract which operations are backed by cuda kernels, as only those have timing information.
        total_cuda_time (_type_): The total cuda execution time, taking into account potential parallel executions.

    Returns:
        float: The largest ratio of total time found in any node.
    """
    largest_ratio_of_total_cuda_time = 0.0
    cuda_kernel_backed_operations = operations_library.getCUDAKernelsBackedOperationNames()

    # Iterate over nodes that have cuda lernels, if node has duration, calculate potential max speedup and store it in the node.
    cuda_vertices = [graph_vertex for graph_vertex in graph.vertices() if graph.vp['op_type'][graph_vertex].lower() in cuda_kernel_backed_operations]
    for vertex in cuda_vertices:
        # Calculate how much of the total cuda time is spent in this node.
        node_duration = graph.vp['duration'][vertex]
        ratio_of_total_cuda_time = node_duration / total_cuda_time
        graph.vp['ratio_of_cuda_time'][vertex] = ratio_of_total_cuda_time
        # Keep track of the largest of these ratios, to prioritize optimization efforst.
        if ratio_of_total_cuda_time > largest_ratio_of_total_cuda_time:
            largest_ratio_of_total_cuda_time = ratio_of_total_cuda_time
        # Calculate theoritical speedup if this node was optimized to an execution time of 0.
        max_speedup = 1 / (1-ratio_of_total_cuda_time)
        graph.vp['max_speedup'][vertex] = max_speedup
    
    return largest_ratio_of_total_cuda_time



def addTimingInformation(
    graph: gt.Graph,
    json_trace_file: str,
    operations_library: operationsLibrary,
    debug: bool = False,
):
    print('Loading json trace file for kernel times...') if debug else None
    cuda_traces = loadJSONCudaTraces(json_trace_file)
    included_traces = 0
    slowest_node_time = 0
    

    graph_vertices = [vertex for vertex in graph.vertices() if graph.vp['op_type'][vertex] != 'input']
    cuda_kernel_backed_operations = operations_library.getCUDAKernelsBackedOperationNames()

    # Iterate over graph nodes, if op_type is not input, a kernel is expected to exist for that operation.
    for vertex in graph_vertices:
        op_type = graph.vp['op_type'][vertex].lower()
        if op_type in cuda_kernel_backed_operations:
            # Look for the first trace with a name that matches the corresponding regex in the cuda kernel mapping
            traces_for_op_type = [trace for trace in cuda_traces if re.match(operations_library.getOperationKernelRegex(op_type), trace.get('name'))]
            if len(traces_for_op_type) == 0:
                # No traces found for this operation type.
                print('Kernel for operation type ', op_type, ' not found: no trace name containing ', operations_library.getOperationKernelRegex(op_type))
                print('Pending Traces: ')
                for trace in cuda_traces:
                    print(trace.get('name'), '\n')
            else:
                # One or more traces for this operation type found. Use the first one and remove it from the list.
                trace = traces_for_op_type[0]
                cuda_traces.remove(trace)
                duration = trace.get('dur')
                # Add start time and duration to the node
                graph.vp['start_time'][vertex] = trace.get('ts')
                graph.vp['duration'][vertex] = duration
                print('\t\tAdding start time ', trace.get('ts'), ' and duration ', duration, ' to the node ', graph.vp['name'][vertex]) if debug else None
                included_traces+=1
                # Update slowest node time. If the current node has longer duration, it is the new slowest node.
                if duration > slowest_node_time:
                    slowest_node_time = duration
        else:
            # Operation type does not have a corresponding kernel. Set color to grey.
            print('Operation type ', op_type, ' does not have corresponding kernel name in mapping.') if debug else None
            graph.vp['fillcolor'][vertex] = '#f0f0f0'

    cuda_busy_time, cuda_wall_time = calculate_cuda_wall_time(graph, debug=debug)
    print('\tCalculated cuda busy time:', convert_microseconds(cuda_busy_time),'and wall time of', convert_microseconds(cuda_wall_time))# if debug else None

    print('Nodes with added timing information: ', included_traces) if debug else None
    largest_ratio_of_total_cuda_time = updateTimeRatiosOfTotal(graph, operations_library, cuda_busy_time)
    # Store largest ratio in graph property
    graph.gp['max_ratio_of_cuda_time'] = largest_ratio_of_total_cuda_time
    print('Slowest operation ratio of total time: ', graph.gp['max_ratio_of_cuda_time']) if debug else None

    # Identify critical path on the graph
    print('\tIdentifying critical path...')
    identifyCriticalPath(graph, debug=debug)

    

    

# Receives a pydot graph and will search for groups of operations performed in the same kernel accoding to the operationLibrary provided. Outputs a new graph with one node per cuda kernel.
def groupOperationsByKernelsWithLibrary(
    graph: gt.Graph,
    operations_library: operationsLibrary,
    embed_docstring: bool = False,
    debug: bool = False,
) -> pydot.Dot:
    next_id = 0 # New graph will have new ids otherwise the new nodes will skip ids.
    grouped_operations_graph = createGraphToolGraph(graph.gp["name"])

    group_starters = operations_library.getStartingOperations()# Group starter operations to have on hand.
    grouped_vertex = [] # List that stores the grouped operations nodes. These operations will be part of grouped nodes and should not be included in the new graph.
    grouped_vertex_to_group_mapping = {} # Store the group where each node in a group is included.
    graphs_vertex_mapping = {} # Store the vertex descriptors of corresponding nodes in both graphs, neccesary to copy edges between nodes with different names due to new contiguous ids in the new graph. indexed by node name in the original graph

    print('Group starters: ', group_starters) if debug else None
    original_graph_vertices = graph.vertices()
    remaining_vertices = [vertex for vertex in graph.vertices()]# Copy to modify while iterating the same list
    # Go over every node in the graph
    print('Starting graph nodes grouping...') if debug else None
    completed_nodes = 0
    total_nodes = len(remaining_vertices)
    for vertex in original_graph_vertices:
        print_progress_bar(completed_nodes, total_nodes)
        vertex_name = graph.vp["name"][vertex]
        print('Node being processed: '+vertex_name) if debug else None
        # Check that the current node is not part of an already added group.
        if vertex not in grouped_vertex:
            # if op is a group starting operation, look ahead for the rest of the group.
            vertex_operation = graph.vp["op_type"][vertex].strip().lower().replace("\"", "")
            if vertex_operation in group_starters:
                print("\tOperation "+vertex_operation+' is a group starter.') if debug else None
                # Iterate over operation groups, for each group which first operation is this one, check for full group present
                largest_group_size = 0
                largest_group_name = None
                for operation_group in operations_library.getGroupedOperations():
                    print(f"\tChecking operation group {operation_group.getName()}:") if debug else None
                    start_time = time.time()
                    if vertex_operation in operation_group.getStartingOperations() and operation_group.isOperationGroupPresentInGraph(graph, [vertex], remaining_vertices, operations_library, debug=debug):
                        # Current operation starts this group and the full group is present. 
                        # Update largest group if full group and current group is larger than previous largest.
                        group_front, nodes_in_group = operation_group.matchNodesForGroup(graph, [vertex], remaining_vertices, operations_library, debug=debug)
                        current_group_length = len(nodes_in_group)
                        print('\t\tGroup lookahead completed for group '+ operation_group.getName() +' at '+str(current_group_length)+' operations.') if debug else None
                        if current_group_length > largest_group_size:
                            # This group is larger than previous ones. Larger operation groups have precedence.
                            if largest_group_name is None:
                                print('\t\tOperation group completed: '+operation_group.getName()+' of size '+str(current_group_length)+', first group found.' ) if debug else None
                            else:
                                print('\t\tOperation group completed: '+operation_group.getName()+' of size '+str(current_group_length)+', larger than the previous one: '+largest_group_name+' at size '+str(largest_group_size)) if debug else None
                            largest_group_name = operation_group.getName()
                            largest_group_size = current_group_length
                            largest_group_vertices = nodes_in_group
                            print('\t\tLargest group nodes:', [graph.vp['name'][lg_vertex] for lg_vertex in largest_group_vertices]) if debug else None
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"\tTime do check group {operation_group.getName()}: {elapsed_time} seconds") if debug else None
                # Add node to new graph. Only if group has been found
                if largest_group_size > 0:
                    nodes_path = [graph.vp['node_path'][vertex] for vertex in largest_group_vertices]
                    new_vertex = addVertexToGraph(grouped_operations_graph, op_type=largest_group_name, op_id=next_id, nodes_path=nodes_path, name=None, **CUDA_OP_STYLE)
                    print('\tAdding grouped operations node to new graph: '+grouped_operations_graph.vp['name'][new_vertex], 'with path:', grouped_operations_graph.vp['name'][new_vertex]) if debug else None
                    next_id+=1
                    # Add nodes in the group to the list of grouped nodes
                    grouped_vertex.extend(largest_group_vertices)# Add names (with id) of nodes in the group to list to skip them thank to the "if node not in grouped_nodes:" check.
                    old_remaining_node_amount = len(remaining_vertices)
                    remaining_vertices = [vertex for vertex in remaining_vertices if vertex not in largest_group_vertices] # remove nodes in the group from the remaining nodes
                    print('\t\tRemaining node amount changes from:', old_remaining_node_amount, 'to', len(remaining_vertices)) if debug else None
                    print('\t\tgrouped_nodes list: '+str(grouped_vertex)) if debug else None
                    # Store which grouped node represents the nodes grouped together for edge recreation
                    for included_vertex in largest_group_vertices:
                        grouped_vertex_to_group_mapping[included_vertex] = new_vertex
                else:
                    # Cannot find any group, add as a normal replicated node
                    print('\tCannot match any of the groups this operation can start, replicating to new graph.') if debug else None
                    new_vertex_properties = get_vertex_properties(graph, vertex)
                    new_vertex_properties['op_id']=next_id
                    new_vertex_properties['name'] = None # Delete vertex name to generate new one with new ids
                    if new_vertex_properties['op_type'] == 'input':
                        new_vertex_properties['op_id']=-1
                    else:
                        next_id+=1
                    new_vertex = addVertexToGraph(grouped_operations_graph, **new_vertex_properties)
                    graphs_vertex_mapping[vertex] = new_vertex
                    if vertex in remaining_vertices:
                        print('\tRemoving node from remaining nodes.') if debug else None
                        remaining_vertices.remove(vertex)
            else:
                # Node does not start any operation and it is not in grouped with another nodes in a grouped operation, copy to new graph.
                print('\tNode', vertex_operation, 'does not start any grouped operation node, replicating to new graph.') if debug else None
                new_vertex_properties = get_vertex_properties(graph, vertex)
                new_vertex_properties['op_id']=next_id
                if new_vertex_properties['op_type'] != 'input':
                    # Only increase next id if the operation is not an input (they have no id)
                    next_id+=1
                    new_vertex_properties['name'] = None # Delete vertex name to generate new one with new ids
                else:
                    # Input operation, override in_od to -1
                     new_vertex_properties['op_id']=-1
                new_vertex = addVertexToGraph(grouped_operations_graph, **new_vertex_properties)
                graphs_vertex_mapping[vertex] = new_vertex
                print('Node to remove:', vertex_name, '\nRemaining nodes:', [graph.vp["name"][rem_vertex] for rem_vertex in remaining_vertices]) if debug else None# +hex(id(node))
                if vertex in remaining_vertices:
                    print('\tRemoving node from remaining nodes.') if debug else None
                    remaining_vertices.remove(vertex) 
        else:
            # Node is part of a group that has been added to the new graph. Do nothing, do not add to new graph nor add new edges.
            print('Node', vertex_name, 'is part of an already formed grouped node.') if debug else None
        completed_nodes+=1
    # for node in graph.get_node_list(): end
    # Clear progress bar
    print(' '*100, end='\r')

    # Replicate edges.
    for edge in graph.edges():
        edge_start = None
        edge_end = None
        # Check source node is last of any group.
        if edge.source() in grouped_vertex_to_group_mapping.keys():
            # The copy of this edge needs to start in a grouped operations node.
            edge_start = grouped_vertex_to_group_mapping[edge.source()]
        elif edge.source() in graphs_vertex_mapping.keys():
            # Source of the edge is a normal (copied) node in the new graph. Should be in new_graph_node_name_mapping
            edge_start = graphs_vertex_mapping[edge.source()]
        # Check destination.
        if edge.target() in grouped_vertex_to_group_mapping.keys():
            # Destination of edge is a new grouped operations node.
            edge_end = grouped_vertex_to_group_mapping[edge.target()]
        elif edge.target() in graphs_vertex_mapping.keys():
            # Source of the edge is a normal (copied) node in the new graph. Should be in new_graph_node_name_mapping
            edge_end = graphs_vertex_mapping[edge.target()]
        # If edge has source and end, add to graph
        if edge_start is not None and edge_end is not None and edge_start != edge_end:
            new_edge = grouped_operations_graph.add_edge(edge_start, edge_end)
            grouped_operations_graph.ep['branch_aggregated_ratio'][new_edge] = -1 # Set branch aggregated ratio to -1 to hide label.



    #return new grouped operations graph.
    return grouped_operations_graph




def GetCudaOnlyPydotGraph(  # noqa: N802
    onnx_graph: GraphProto,
    operations_library: operationsLibrary,
    trace_file: str = None,
    simplegraph_svg: str = None,
    name: str | None = None,
    rankdir: str = "TB",
    embed_docstring: bool = False,
    debug: bool = False,
) -> pydot.Dot:
    gt_graph = createGraphToolGraph(name=name)
    input_generators: dict[str, gt.Vertex]  = {}# Stores which node generated the input.

    input_creating_op_types = [operation.getName() for operation in operations_library.getInputCreatingOperations()]
    
    print(f'Creating graph from ONNX representation with {len(onnx_graph.node)} nodes... ', end='')
    start_time = time.time()
    for op_id, op in enumerate(onnx_graph.node):
        #if op.op_type.lower() not in input_creating_op_types:
        if op.op_type.lower() not in input_creating_op_types:# Version with all operations in the model. Needed if only one layer of the model is provided, the names change in that case.
        #if op.op_type.lower() not in input_creating_op_types and 'layer' in op.name:# Version only with inside layers
            # Operationshould be included in the graph, add to graph.
            op_vertex = addVertexToGraph(gt_graph, node_path=op.name, op_type=op.op_type, op_id=op_id, **CUDA_OP_STYLE)
            # Add edges to previous op. Iterate inputs and search for input generators for those inputs
            for input_name in op.input:
                previous_vertex = None
                if len(input_generators)==0: 
                    # First op, it has inputs but not generated by other nodes
                    # Create new graph node, initial input
                    input_vertex = addVertexToGraph(gt_graph, name=_escape_label(input_name), **INPUT_STYLE)
                    input_generators[input_name] = input_vertex
                    
                # Get input generator from dict
                if previous_vertex is None and input_name in input_generators:
                    previous_vertex = input_generators[input_name]
                # Add edges from previous node to current
                if previous_vertex is not None:
                    new_edge = gt_graph.add_edge(previous_vertex, op_vertex)
                    gt_graph.ep['branch_aggregated_ratio'][new_edge] = -1 # Set branch aggregated ratio to -1 to hide label.
            # Add generated outputs to pydot_input_generators
            for output_name in op.output:
                input_generators[output_name] = op_vertex
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{elapsed_time:.3f} seconds")

    # Calculate how many parallel paths each node has. 0 in a vertex means there are no other paths in the graph to reach later vertices.
    # Locate graph root and end, vertex with in-degree and out-degree to 0, respectively.
    graph_root = None
    for vertex in gt_graph.vertices():
        if vertex.in_degree() == 0:
            graph_root = vertex

    if graph_root is None:
        print(f'No graph root detected on {gt_graph.num_vertices()} nodes.')
    
    current_parallel_paths = graph_root.out_degree()-1
    next_vertices_to_explore = [graph_root]
    gt_graph.vp['parallel_paths'][graph_root] = 0
    while len(next_vertices_to_explore) > 0:
        following_vertices = []
        # Only expand the vertex with lowest id, copy the rest to next step
        lowest_id = None
        # Locate lowest id vertex
        for vertex in next_vertices_to_explore:
            if lowest_id is None or gt_graph.vp['op_id'][vertex] < lowest_id:
                lowest_id = gt_graph.vp['op_id'][vertex]
        # Go over next_vertices_to_explore again, expanding the lowest id one and copying the rest
        for vertex in next_vertices_to_explore:
            if gt_graph.vp['op_id'][vertex] == lowest_id:
                # Expand lowest id vertex
                for next_vertex in vertex.out_neighbors():
                    if next_vertex not in following_vertices:
                        following_vertices.append(next_vertex)
            elif vertex not in following_vertices:
                # Add only if not already in the list
                following_vertices.append(vertex)
        
        # Update parallel paths value for all following vertices 
        current_parallel_paths = len(following_vertices)
        for vertex in following_vertices:
            gt_graph.vp['parallel_paths'][vertex] = current_parallel_paths-1

        # Step forward in the graph:
        next_vertices_to_explore = following_vertices

    if simplegraph_svg is not None:
        print_safe_graph=sanitize_graph(exportGraphToPydot(gt_graph, onnx_graph.name, rankdir=rankdir, debug=debug)) 
        print_safe_graph.write_svg(simplegraph_svg)

    # Go over pydot graph and group operations according to multi-operation kernels.
    print('Grouping operations to match CUDA kernels...')
    start_time = time.time()
    grouped_graph = groupOperationsByKernelsWithLibrary(gt_graph, operations_library, embed_docstring=embed_docstring, debug=debug)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\tGrouping completed in {elapsed_time:.3f} seconds")
    
    #  Update graph nodes with duration and speedup information.
    if trace_file is not None:
        print('Adding timing information for max speedups and critical path...')
        start_time = time.time()
        addTimingInformation(grouped_graph, json_trace_file=trace_file, operations_library=operations_library, debug=debug)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\tTime information included in {elapsed_time:.3f} seconds")
    else:
        print("No trace file provided, graph with no execution time information")


    #return pydot_graph
    return grouped_graph

def GetSimplerPydotGraph(  # noqa: N802
    graph: GraphProto,
    name: str | None = None,
    rankdir: str = "TB",
    embed_docstring: bool = False,
) -> pydot.Dot:
    node_producer = GetOpNodeProducer(embed_docstring=embed_docstring, **CUDA_OP_STYLE)
    pydot_graph = pydot.Dot(name, rankdir=rankdir)
    pydot_input_generators: dict[str, pydot.Node]  = {}# Stores which node generated the input.
    
    for op_id, op in enumerate(graph.node):
        op_node = node_producer(op, op_id, 0)
        pydot_graph.add_node(op_node)
        # Add edges to previous op. Iterate inputs and search for input generators for those inputs
        for input_name in op.input:
            previous_node = None
            if len(pydot_input_generators)==0: 
                # First op, it has inputs but not generated by other nodes
                # Create new graph node, initial input
                input_node = pydot.Node(
                    _escape_label(input_name),
                    label=_escape_label(input_name),
                    **INPUT_STYLE,
                )
                # Add simple input to graph
                pydot_graph.add_node(input_node)
                # Add to input generators, an input generator that is simply the input itself
                pydot_input_generators[input_name] = input_node
                
            # Get input generator form dict
            if previous_node is None and input_name in pydot_input_generators:
                previous_node = pydot_input_generators[input_name]
            # Add edges from previous node to current
            if previous_node is not None:
                pydot_graph.add_edge(pydot.Edge(previous_node, op_node))
            
        # Add generated outputs to pydot_input_generators
        for output_name in op.output:
            pydot_input_generators[output_name] = op_node
            
    
    return pydot_graph

def exportGraphToPydot(
    source_graph: gt.Graph,
    graph_name: str | None = None,
    rankdir: str = "TB",
    debug: bool = False,
):
    # Create new pydot graph
    pydot_graph = pydot.Dot(graph_name, rankdir=rankdir)
    vertex_to_node_mapping = {}
    largest_ratio_of_total_cuda_time = source_graph.gp['max_ratio_of_cuda_time']

    # Add all vertices from source graph, with all atributes for each vertex
    for vertex in source_graph.vertices():
        properties = get_vertex_properties(source_graph, vertex)

        pydot_node = pydot.Node( **properties)# source_graph.vp['name'][vertex],
        updatePydotNodeStyle(pydot_node, largest_ratio_of_total_cuda_time)# Update node style for exporting.
        updatePydotNodeURL(pydot_node) # Generate javascript for clicing on the nodes.
        pydot_graph.add_node(pydot_node)
        vertex_to_node_mapping[vertex] = pydot_node

    # Add all edges from the source graph, with all attributes for each edge
    for edge in source_graph.edges():
        properties = get_edge_properties(source_graph, edge)
        edge = pydot.Edge(vertex_to_node_mapping[edge.source()], vertex_to_node_mapping[edge.target()], **properties)
        updatePydotEdgeStyle(edge) # Update styling to show criticality.
        pydot_graph.add_edge(edge)
    

    return pydot_graph
    # Export to svg file
    



def GetFullPydotGraph(  # noqa: N802
    graph: GraphProto,
    name: str | None = None,
    rankdir: str = "TB",
    embed_docstring: bool = False,
) -> pydot.Dot:
    node_producer = GetOpNodeProducer(embed_docstring=embed_docstring, **CUDA_OP_STYLE)
    pydot_graph = pydot.Dot(name, rankdir=rankdir)
    pydot_nodes: dict[str, pydot.Node] = {}
    pydot_node_counts: dict[str, int] = defaultdict(int)
    for op_id, op in enumerate(graph.node):
        op_node = node_producer(op, op_id)
        pydot_graph.add_node(op_node)
        for input_name in op.input:
            if input_name not in pydot_nodes:
                input_node = pydot.Node(
                    _escape_label(input_name + str(pydot_node_counts[input_name])),
                    label=_escape_label(input_name),
                    **INPUT_STYLE,
                )
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            pydot_graph.add_node(input_node)
            pydot_graph.add_edge(pydot.Edge(input_node, op_node))
        for output_name in op.output:
            if output_name in pydot_nodes:
                pydot_node_counts[output_name] += 1
            output_node = pydot.Node(
                _escape_label(output_name + str(pydot_node_counts[output_name])),
                label=_escape_label(output_name),
                **INPUT_STYLE,
            )
            pydot_nodes[output_name] = output_node
            pydot_graph.add_node(output_node)
            pydot_graph.add_edge(pydot.Edge(op_node, output_node))
    return pydot_graph


def generateUpdatedTimingInformation(
    gtgraph: gt.Graph,
    onnx_graph: GraphProto,
    operations_library: operationsLibrary,
    speedups_json_file: str,
    svg:str = None,
    rankdir: str = "TB",
    debug: bool = False,
) -> pydot.Dot:

    def loadSpeedupsJSON(
        speedups_json_file: str,
        debug: bool = False,
    ):
        returned_speedups = {}
        # Load JSON data from a file
        with open(speedups_json_file) as f:
            data = json.load(f)

        # Iterate over speedup entries and load op_name and speedup value.
        for speedup_entry in data['speedups']:
            if 'op_id' in speedup_entry:
                # Speedup applies to only one operation. Store in map with key ID:op_id
                returned_speedups['ID:'+speedup_entry.get('op_id')] = speedup_entry.get('speedup')
                print(f'Loading speedup of {speedup_entry.get("speedup")} for ID:{speedup_entry.get("op_id")}')  if debug else None
            if 'op_type' in speedup_entry:
                # Speedup applies to only all operations of th etype. Store in map with key TYPE:op_type
                returned_speedups['TYPE:'+speedup_entry.get('op_type')] = speedup_entry.get('speedup')
                print(f'Loading speedup of {speedup_entry.get("speedup")} for TYPE:{speedup_entry.get("op_type")}')  if debug else None
            # No id or type provided, ignore entry.
        
        return returned_speedups
    # loadSpeedupsJSON end.

    print('Generating new timing information using the provided speedups...')

    updated_times_graph = gtgraph.copy()

    # Load json with speedup descriptions.
    print('Loading speedup descriptions...')  if debug else None
    speedups_to_apply = loadSpeedupsJSON(speedups_json_file, debug=debug)

    # Iterate over graph, copying nodes to new graph and updating the durations
    for graph_vertex in updated_times_graph.vertices():
        # Check if id is in speedups keys with ID:prefix
        if gtgraph.vp['op_id'][graph_vertex] != -1 and ('ID:'+str(gtgraph.vp['op_id'][graph_vertex])) in speedups_to_apply.keys():
            # Found speedup to apply to this operations by id
            new_duration = gtgraph.vp['duration'][graph_vertex]/speedups_to_apply['ID:'+str(gtgraph.vp['op_id'][graph_vertex])]
            print('Reduced operation', gtgraph.vp['name'][graph_vertex], 's duration from', gtgraph.vp['duration'][graph_vertex], 'to', new_duration) if debug else None
            updated_times_graph.vp['duration'][graph_vertex] = new_duration
        elif gtgraph.vp['op_type'][graph_vertex] != "" and ('TYPE:'+gtgraph.vp['op_type'][graph_vertex]) in speedups_to_apply.keys():
            # Found speedup to apply to all operations of this type
            new_duration = gtgraph.vp['duration'][graph_vertex]/speedups_to_apply['TYPE:'+gtgraph.vp['op_type'][graph_vertex]]
            print('Reduced operation', gtgraph.vp['name'][graph_vertex], 's duration from', gtgraph.vp['duration'][graph_vertex], 'to', new_duration) if debug else None
            updated_times_graph.vp['duration'][graph_vertex] = new_duration


    # Update data derived from timing. Total cuda time, ratios and critical path
    print('Updating time-derived information...')  if debug else None
    original_cuda_busy_time, original_cuda_wall_time = calculate_cuda_wall_time(gtgraph, debug=debug)
    updated_cuda_busy_time, updated_cuda_wall_time = calculate_cuda_wall_time(updated_times_graph, debug=debug)
    # Updated wall time will be the same unless the last node was improved. Calculate proper updated wall time using the improvement in busy time.
    # It makes the assumption that the wait times are the same but everything after an sped up node can start sooner.
    busy_time_improvement = original_cuda_busy_time - updated_cuda_busy_time
    updated_cuda_wall_time = original_cuda_wall_time-busy_time_improvement
    # Show new timing summary.
    print('\tUpdated cuda busy time:', convert_microseconds(updated_cuda_busy_time),'and estimated best-case-scenario for updated wall time of', convert_microseconds(updated_cuda_wall_time))# if debug else None

    largest_ratio_of_total_cuda_time = updateTimeRatiosOfTotal(updated_times_graph, operations_library, updated_cuda_busy_time)
    updated_times_graph.gp['max_ratio_of_cuda_time'] = largest_ratio_of_total_cuda_time
    identifyCriticalPath(updated_times_graph, debug=debug)

    # Export updated data to JSON
    print('Exporting data to JSON for model-explorer...')  if debug else None
    exportTimingDataToJSON(updated_times_graph, onnx_graph, "time_ratio_node_data_with_speedups.json", debug=debug)

    if svg is not None:
        # Export updated graph to svg
        print('Exporting updated graph to svg...')  if debug else None
        print_safe_graph=sanitize_graph(exportGraphToPydot(updated_times_graph, onnx_graph.name, rankdir=rankdir, debug=debug))
        print_safe_graph.write_svg(svg[:-4]+'_with_speedups.svg')
    return 


def sanitize_graph(
    graph: pydot.Dot,
    ) -> pydot.Dot:

    sanitized_graph = pydot.Dot(graph.get_name(), rankdir=graph.get_rankdir())
    node_sanitizer = GetOpNodeSanitizer()
    edge_sanitizer = GetEdgeSanitizer()
    #print('Sanitizing graph for export...')
    for node in graph.get_node_list():
        sanitized_graph.add_node(node_sanitizer(node))
    
    for edge in graph.get_edge_list():
        sanitized_graph.add_edge(edge_sanitizer(edge))

    return sanitized_graph




def main() -> None:
    print('Network performance analyser V1.0')
    parser = argparse.ArgumentParser(description="ONNX net drawer")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The input protobuf file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output JSON file for Model Explorer.",
    )
    parser.add_argument(
        "--traces",
        type=str,
        default=None,
        help="json file with trace events of one execution to add execution times to graph.",
    )
    parser.add_argument(
        "--rankdir",
        type=str,
        default="TB",
        help="The rank direction of the pydot graph.",
    )
    parser.add_argument(
        "--embed_docstring",
        action="store_true",
        help="Embed docstring as javascript alert. Useful for SVG format.",
    )
    parser.add_argument(
        "--svg",
        type=str,
        default=None,
        help="Export graph in SVG format to the indicated file.",
    )
    parser.add_argument(
        "--speedups",
        type=str,
        default=None,
        help="JSON with achievable speedups for different ops to generate updated timing information.",
    )
    parser.add_argument(
        "--fullgraph",
        type=str,
        default=None,
        help="SVG output file that shows all of the ONNX operations without grouping for CUDA kernels.",
    )
    parser.add_argument(
        "--simplegraph",
        type=str,
        default=None,
        help="SVG output file that shows only the ONNX operations where grouping will be performed but before the grouping step.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    args = parser.parse_args()
    model = ModelProto()
    with open(args.input, "rb") as fid:
        content = fid.read()
        model.ParseFromString(content)

    # If fullgraph option was given, create a simplified (not input/output nodes) graph with all ONNX operations, without grouping nodes according to CUDA kernels
    if args.fullgraph is not None:
        full_onnx_graph = GetSimplerPydotGraph(
            model.graph,
            name=model.graph.name,
            rankdir=args.rankdir,
        )
        print('Creating full ONNX graph with all operations... ', end="")
        start_time = time.time()
        full_onnx_graph.write_svg(args.fullgraph)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{elapsed_time:.4f} seconds")
        

    operationLib = operationsLibrary('./repositorios/MLDevOps/amdahl4ML/operationsAndGroups.xml')

    cuda_graph = GetCudaOnlyPydotGraph(
        model.graph,
        operationLib,
        trace_file=args.traces,
        simplegraph_svg = args.simplegraph,
        name=model.graph.name,
        rankdir=args.rankdir,
        debug=args.debug,
    )

    if args.traces is not None:
        #Export timing and critical path information to JSON 
        exportTimingDataToJSON(cuda_graph, model.graph, args.output, debug=args.debug)

    if args.speedups is not None:
        # Use given speedups to generate a new timing and critical path custom node data
        generateUpdatedTimingInformation(cuda_graph, model.graph, operationLib, args.speedups, args.svg, rankdir=args.rankdir, debug=args.debug)

    if args.svg is not None:
        #exportGraphToPydot(cuda_graph, args.svg, model.graph.name, rankdir=args.rankdir, debug=args.debug)
        # Sanitize node attributes fro printing
        print_safe_graph=sanitize_graph(exportGraphToPydot(cuda_graph, model.graph.name, rankdir=args.rankdir, debug=args.debug)) 
        print_safe_graph.write_svg(args.svg)

    

if __name__ == "__main__":
    main()
