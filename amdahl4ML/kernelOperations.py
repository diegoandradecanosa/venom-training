

import xml.etree.ElementTree as ET
import graph_tool as gt
from typing import Tuple


class operationDescription:
    """
    Superclass for describing relations between operations. Use subclasses.
    """

    def getStartingOperations(self)->list[str]:
        raise NotImplementedError("Subclass must implement abstract method")
    
    @classmethod
    def buildFromXMLNode(cls, node, operationsLibrary, debug:bool = False):

        # Get node's children
        if node.tag == 'nodes':
            # Node is the entire node description for the group. Only one sub element, sequential, branching, miscellaneous, single node or empty branch.
            # Element to process is the first and only children of the node.
            element = list(node)[0]
        else:
            # Node is not the root operation description, method called for recursively build the descrition. Element to process is the node itself.
            element = node
        print('Entry point tag:', node.tag, ' tag to process:', element.tag) if debug else None
        # Check type of node by the tag, redirect to subclass method.
        if element.tag == 'node':    
            return singleOperation.buildFromXMLNode(element, operationsLibrary)
        elif element.tag == 'sequential':
            return sequentialOperations.buildFromXMLNode(element, operationsLibrary)
        elif element.tag == 'branching':
            return branchingOperations.buildFromXMLNode(element, operationsLibrary)
        elif element.tag == 'branchedStart':
            return branchedStartOperations.buildFromXMLNode(element, operationsLibrary)
        elif element.tag == 'emptyBranch':
            return emptyBranch.buildFromXMLNode(element, operationsLibrary)
        elif element.tag == 'miscellaneousNodes':
            return miscellaneousOperationsSet.buildFromXMLNode(element, operationsLibrary)
        elif element.tag == 'crossbranchjump':
            return crossBranchJump.buildFromXMLNode(element, operationsLibrary)
        else:
            raise TypeError("Operation description tag not supported: "+element.tag)

    def stringSummary(self, tablevel: int) -> str:
        raise NotImplementedError("Subclass must implement abstract method")


    def isOperationNextInGraph(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> bool:
        """
            Returns whether or not this operation is present in the graph starting with the operations in next_vertices_in_graph.

        Args:
            graph (gt.Graph): The complete operation graph.
            next_vertices_in_graph (list[gt.Vertex]): Where in the graph the operations starts, the first operations still to be matched.
            operationsLibrary (_type_): The library of possible operations to match.
            debug (bool, optional): Whether or not to print debug information. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            bool: _description_
        """
        raise NotImplementedError("Subclass must implement abstract method")
    
    def advanceOperationFront(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> Tuple[list[gt.Vertex], list[gt.Vertex]]:
        """
        Traverses the graph and returns a tuple of the nodes after this operation (the next step efter this operation) and the nodes included in this operation (the nodes navigated through)

        Args:
            graph (gt.Graph): The graph to traverse.
            next_vertices_in_graph(list[gt.Vertex]): The next nodes in the graph. Marks the current point in the graph.
            vertices_not_already_included(list[gt.Vertex]): The nodes of the graph still not included in some group or included by themselves.
            operationsLibrary(operationsLibrary): The library of operations supported.
            debug(bool): Whether or not to print debug information. Defaults to False.
        Returns:
            Tuple[list[gt.Vertex], list[gt.Vertex]] a tuple of [new_next_nodes, included_nodes]. 
                The first element in the node is a list of the next nodes when this operation is matched, what comes after this operation.
                The second element is the graph nodes that are included in this operation.
        Raises:
            NotImplementedError: If this method is called in the abstract superclass.
        """
        raise NotImplementedError("Subclass must implement abstract method")

class genericOperationType:
    """
    Represents a generic operation type.

    Attributes
    ----------
    name : str
        The name (label) of the operation to be shown in graphs.

    Methods
    -------

    Notes
    -----
    Use of subclasses is recommended.
    """
    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name
    
class miscelaneousOperationType(genericOperationType):

    def __init__(self, name):
        super().__init__(name)

    
class hiddenOperationType(genericOperationType):

    def __init__(self, name):
        super().__init__(name)


class kernelOperationType(genericOperationType):
    """
    Graph operation representing an operation that has a cuda kernel that performs the needed work. 

    Attributes:
        name: str

    """

    def __init__(self, name, kernelRegex ):
        super().__init__(name)
        self.kernelRegex = kernelRegex
    
    def getKernelSignatureRegex(self):
        return self.kernelRegex
        
    def getStartingOperations(self) -> list[str]:
        raise NotImplementedError("Subclass must implement abstract method")


class simpleOperationType(kernelOperationType):
    """
    Operation type that has a direct correlation between onnx representation and cuda kernel.

    Use when one CUDA kernel performs exactly one ONNX operation.

    Attributes:
        onnxName (str): the name of the operation in the ONNX representation, the type of operation.
    """

    def __init__(self, name:str, kernelRegex:str, onnxName:str ):
        """
        Constructs the necessary attributes for a simpleOperationType object.

        Args:
            name (str): The name of the operation type, seen as the kernel itself.
            kernelRegex (str): Regex describing the signature of the kernel call.
            onnxName (str): The name of the ONNX operation this kernel performs.
        """
        super().__init__(name , kernelRegex)
        self.onnxName = onnxName

    
    def getStartingOperations(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return [self.onnxName]

class operationGroup(kernelOperationType):
    """
    Operation type that executes multiple ONNX operations with the same kernel.

    """
    def __init__(self, name:str, kernelRegex:str, groupedOperations: operationDescription ):
        """_summary_

        Args:
            name (str): The name of the operation group, seen as the kernel itself.
            kernelRegex (str): Regex describing the signature of the kernel call.
            groupedOperations (list): List of operations that are performed by the same kernel.
        """
        super().__init__(name , kernelRegex)
        self.operationDescription = groupedOperations

    def getStartingOperations(self):
        return self.operationDescription.getStartingOperations()
    
    def isOperationGroupPresentInGraph(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> bool:
        
        # Use operation description method to check the group is present
        print('Checking for', self.name, ' operation group:') if debug else None
        return self.operationDescription.isOperationNextInGraph(graph, next_vertices_in_graph=next_vertices_in_graph, vertices_not_already_included=vertices_not_already_included, operationsLibrary=operationsLibrary, debug=debug)

    def matchNodesForGroup(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> Tuple[list[gt.Vertex], list[gt.Vertex]]:
        # Use operation description method
        return self.operationDescription.advanceOperationFront(graph, next_vertices_in_graph, vertices_not_already_included, operationsLibrary, debug=debug)


class operationsLibrary:

    def __init__(self, sourceXML:str, debug:bool = False):
        self.sourceXML = sourceXML
        # Initialize operation types list.
        self.miscellaneousOperations = {}
        self.hiddenOperations = {}
        self.singleOperations = {}
        self.groupedOperations = {}
        self.loadFromXML(self.sourceXML, debug=debug)
        print(f'Operations library loaded {len(self.miscellaneousOperations)+len(self.hiddenOperations)+len(self.singleOperations)+len(self.groupedOperations)} operations, of which {len(self.singleOperations)+len(self.groupedOperations)} will create nodes.')

    def getMiscellaneousOperations(self):
        return self.miscellaneousOperations.values()
        
    def getHiddenOperations(self):
        return self.hiddenOperations.values()
    
    def getSingleOperations(self):
        return self.singleOperations.values()

    def getGroupedOperations(self):
        return self.groupedOperations.values()
    
    def getStartingOperations(self):
        group_starters = []
        for group in  self.groupedOperations.values():
            for starting_op in group.getStartingOperations():
                if starting_op not in group_starters:
                    group_starters.append(starting_op)

        return group_starters

    def getInputCreatingOperations(self):
        """
            Returns the set of operations that create inputs, source node on the operation graph.
        """
        return self.getHiddenOperations()

    def loadFromXML(self, xmlFile: str, debug:bool = False)->list[genericOperationType]:

        print('Loading operations library from file: ', xmlFile) if debug else None
        # Parse xml file
        tree = ET.parse(xmlFile)
        root = tree.getroot()
        # Load all miscellaneous ops
        for node in root.findall('miscNode'):
            self.miscellaneousOperations[node.get('name')] = miscelaneousOperationType(node.get('name'))
        # Load hidden operations
        for node in root.findall('hiddenOperation'):
            self.hiddenOperations[node.get('name')] = hiddenOperationType(node.get('name'))
        # Load all single operations
        for node in root.findall('singleOperation'):
            # name:str, kernelRegex:str, onnxName:str
            self.singleOperations[node.get('nodeName')] = simpleOperationType(node.get('name'), node.get('kernelNameRegex'), node.get('nodeName'))
        # Load all grouped operations
        for node in root.findall('OperationGroup'):
            # Use operationDescription's method to get the list of operations grouped. 
            print('Creating grouped operations:', node.find('name').text) if debug else None
            self.groupedOperations[node.find('name').text] = operationGroup(node.find('name').text, node.find('kernelNameRegex').text, operationDescription.buildFromXMLNode(node.find('nodes'), self ))

    def clearOperationsLists(self):
        """
        Empty the operation library, removing all types of operations.
        """
        self.miscellaneousOperations.clear()
        self.hiddenOperations.clear()
        self.singleOperations.clear()
        self.groupedOperations.clear()

    def stringSummary(self) -> str:

        result = 'Operation library:\n'
        result += '\t'+str(len(self.miscellaneousOperations))+' Misc ops, '+ str(len(self.hiddenOperations))+' hidden ops, '+str(len(self.singleOperations))+' single ops and '+str(len(self.groupedOperations))+' grouped ops.\n'
        for misc_op in self.miscellaneousOperations:
            result += '\tMiscellaneous Operation: '+misc_op.name+'\n'
        for hidden_op in self.hiddenOperations:
            result += '\tHidden Operation: '+hidden_op.name+'\n'
        for single_op in self.singleOperations:
            result += '\tSingle Operation: '+single_op.name+' matched to ONNX op '+single_op.onnxName+'\n'
        for grouped_op in self.groupedOperations:
            result += '\tGroupedOperation: '+grouped_op.name+' described as: '+grouped_op.operationDescription.stringSummary(tablevel=1)+'\n'
        return result

    def getOperationKernelRegex(self, operationName:str) -> str:
        if operationName in self.singleOperations.keys():
            return self.singleOperations[operationName].getKernelSignatureRegex()
        if operationName in self.groupedOperations.keys():
            return self.groupedOperations[operationName].getKernelSignatureRegex()
        # Not in single operations nor grouped operations. Does not have a kernel assocciated
        return None

    def getCUDAKernelsBackedOperationNames(self):
        operationNames = []
        operationNames.extend(self.singleOperations.keys())
        operationNames.extend(self.groupedOperations.keys())
        return operationNames

class singleOperation(operationDescription):

    def __init__(self, operationName:str):
        self.operationName=operationName

    def getStartingOperations(self)->list[str]:
        return[self.operationName]

    @classmethod
    def buildFromXMLNode(cls, node, operationsLibrary):
        # Get node text, add single operations based on the name.
        operation_name = node.text
        return singleOperation(operation_name)
    
    def __str__(self):
        return 'SOp('+self.operationName+')'
    
    def stringSummary(self, tablevel: int) -> str:
        return ('\t'*tablevel)+self.operationName
    
    def isOperationNextInGraph(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> bool:
        
        if len(next_vertices_in_graph) == 0:
            print('[OPc] \t\tTesting for single operation', self.operationName, ' with no vertices.') if debug else None
            return False
        print('[OPc] \t\tTesting for single operation', self.operationName, ' with vertex ', graph.vp["op_type"][next_vertices_in_graph[0]] ) if debug else None
        # Return whether or not the next vertex in the graph has the correct op_type
        return graph.vp["op_type"][next_vertices_in_graph[0]].strip().lower().replace("\"", "") == self.operationName.strip().lower().replace("\"", "")

    def advanceOperationFront(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> Tuple[list[gt.Vertex], list[gt.Vertex]]:
        # Get outgoing edges form the current node. 
        new_next_vertices = list(next_vertices_in_graph[0].out_neighbors())
        # Return all of the nodes outgoing edges point to, and the first operation on the given next nodes list.
        print('[OPa] ADVANCED SINGLE OPERATION TO NEXT NODES:', [graph.vp['name'][vertex] for vertex in new_next_vertices])  if debug else None
        return new_next_vertices, [next_vertices_in_graph[0]]




class sequentialOperations(operationDescription):

    def __init__(self, operations:list[operationDescription]):
        self.operations=operations

    def getStartingOperations(self)->list[str]:
        if len(self.operations)>0 and self.operations[0] is not None:
            return self.operations[0].getStartingOperations()
        else:
            return []
    @classmethod
    def buildFromXMLNode(cls, node, operationsLibrary):
        # Return a list of operations, using the abstract class method to correctly instance each subelement
        return sequentialOperations([operationDescription.buildFromXMLNode(subnode, operationsLibrary) for subnode in list(node)])

    def stringSummary(self, tablevel: int) -> str:
        summary = ('\t'*tablevel)+'Sequential:\n'
        for operation in self.operations:
            summary += operation.stringSummary(tablevel+1)+'\n'
        return summary
    
    def __str__(self):
        result = 'Sequence: '
        for operation in self.operations:
            result += operation.__str__()+', '
        return result[:-2]

    def isOperationNextInGraph(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> bool:
        # For every op in the sequence, check that it is the next one in order, navigating the graph
        print('[SQc] Checking operation:', self.__str__(), 'on nodes', [graph.vp['name'][vertex] for vertex in next_vertices_in_graph]) if debug else None

        if len(next_vertices_in_graph)!=1:
            # Sequential node does not support multiple next nodes at the same time. It is not this operation group, but one with branching paths.
            print('[SQc] Multiple front nodes open for sequential operations: error.') if debug else None
            return False
        sequence_index = 0
        next_expected_op = self.operations[sequence_index] # Next expected operation is the first in the sequence
        valid_sequence = True
        while valid_sequence and sequence_index<len(self.operations):
            # Check next node
            print('[SQc]\tChecking step in sequence:', next_expected_op, 'with next_nodes', [graph.vp['name'][vertex] for vertex in next_vertices_in_graph]) if debug else None
            print('[SQc] calling isOperationNextInGraph...') if debug else None
            valid_sequence = next_expected_op.isOperationNextInGraph(graph, next_vertices_in_graph, vertices_not_already_included, operationsLibrary=operationsLibrary, debug=debug)
            if valid_sequence:
                print('[SQc]\tSequential step matched, moving to next in sequence...') if debug else None
                # Advance front to next operation
                print('[SQc] calling advanceOperationFront...') if debug else None
                next_nodes, skipped_ops = self.operations[sequence_index].advanceOperationFront(graph, next_vertices_in_graph, vertices_not_already_included, operationsLibrary=operationsLibrary, debug=debug)
                sequence_index+=1
                if sequence_index<len(self.operations):
                    next_expected_op = self.operations[sequence_index] # Search for the following operation next.
                    next_vertices_in_graph = next_nodes
            else:
                # Next operations are not valid for this sequence.
                print('[SQc]\t\t\tSequence not found' ) if debug else None
                return False

        # While loop ended. Return valid_sequence, if it finishes because it is not a valid sequence, return that false. If it stopped because sequence_index>=len(self.operations), 
        # return true since the end of the group has been reached and all operations are valid.
        print('[SQc] CHECK FOR SEQUENCE:', valid_sequence) if debug else None
        return valid_sequence

    def advanceOperationFront(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> Tuple[list[gt.Vertex], list[gt.Vertex]]:
        # Assumes the full operation group is present and valid. Call this method only if isOperationNextInGraph returns True.
        print("[SQa]\tAdvancing sequential nodes...") if debug else None
        next_vertices = next_vertices_in_graph# If no operations are advanced, return the same next vertices
        included_nodes: list[gt.Vertex] = []
        sequence_index = 0
        while sequence_index<len(self.operations):
            print('[SQa] calling advanceOperationFront...') if debug else None
            next_vertices, op_vertices = self.operations[sequence_index].advanceOperationFront(graph, next_vertices, vertices_not_already_included, operationsLibrary, debug=debug)
            sequence_index+=1
            for vertex in op_vertices:
                if vertex not in included_nodes:
                    included_nodes.append(vertex)
            print('[SQa]\t\tStep advanced:', self.operations[sequence_index-1], 'advanced nodes:', [graph.vp['name'][vertex] for vertex in op_vertices], 'next nodes:', [graph.vp['name'][vertex] for vertex in next_vertices]) if debug else None
        print('[SQa] SEQUENCE OF OPERATIONS ADVANCED TO NEXT NODES:', [graph.vp['name'][vertex] for vertex in next_vertices])  if debug else None
        return next_vertices, included_nodes


class emptyBranch(operationDescription):
    """
    Represents a branch with no operations. Intended for fully describing a group where some data skips operations.

    Methods:
        getStartingOperations (self): returns the name of the first operations. Since the empty branches do not have any operations, it returns and empty list.
    """
    def getStartingOperations(self)->list[str]:
        return []

    @classmethod
    def buildFromXMLNode(cls, node, operationsLibrary):
        return emptyBranch()

    def stringSummary(self, tablevel: int) -> str:
        return ('\t'*tablevel)+'Empty branch'
    
    def __str__(self):
        return 'Empty branch'
    
    def isOperationNextInGraph(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> bool:
        """
        Empty branches always are present since they have no nodes that can be not present.
        Args:
            graph (gt.Graph): _description_
            non_grouped_nodes_remaining (list[gt.Vertex]): _description_
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            bool: True.
        """ 
        print('[EBc]\tCHECK FOR EMPTY BRANCH: True') if debug else None
        return True

    def advanceOperationFront(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> Tuple[list[gt.Vertex], list[gt.Vertex]]:
        print('[EBa]\tADVANCED EMPTY BRANCH TO', [graph.vp['name'][vertex] for vertex in next_vertices_in_graph]) if debug else None
        return next_vertices_in_graph, []





class branchedStartOperations(operationDescription):

    def __getOpenFront__(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        debug: bool = False
    )->list[gt.Vertex]:
        """Obtains the not yet processed nodes that start all branches of this group

        Args:
            graph (gt.Graph): _description_
            vertices_not_already_included (list[gt.Vertex]): _description_

        Returns:
            list[gt.Vertex]: _description_
        """
        # Identify the area in the graph the front can be in, using the next_vertices_in_graph as starting point. 
        lowest_parallel_paths_vertex = None
        highest_parallel_paths_vertex = None
        for vertex in next_vertices_in_graph:
            if lowest_parallel_paths_vertex is None or graph.vp['parallel_paths'][vertex] < graph.vp['parallel_paths'][lowest_parallel_paths_vertex]:
                lowest_parallel_paths_vertex = vertex
            if highest_parallel_paths_vertex is None or graph.vp['parallel_paths'][vertex] < graph.vp['parallel_paths'][highest_parallel_paths_vertex]:
                highest_parallel_paths_vertex = vertex
        # Navigate back until vertex with no parallel paths is found. 
        while graph.vp['parallel_paths'][lowest_parallel_paths_vertex] > 0:
            vertex_in_neighbors_iterator = lowest_parallel_paths_vertex.in_neighbors()
            lowest_parallel_paths_vertex = None # Delete to be able to go back in all in_neighbors, even if they have more parallel_paths than current vertex 
            for following_vertex in vertex_in_neighbors_iterator:
                if lowest_parallel_paths_vertex is None or graph.vp['parallel_paths'][following_vertex] < graph.vp['parallel_paths'][lowest_parallel_paths_vertex]:
                    lowest_parallel_paths_vertex = following_vertex

        # Navigate forward until vertex with no parallel paths is found. 
        while graph.vp['parallel_paths'][highest_parallel_paths_vertex] > 0:
            vertex_out_neighbors_iterator = highest_parallel_paths_vertex.out_neighbors()
            highest_parallel_paths_vertex = None # Delete to be able to go forward in all in_neighbors, even if they have more parallel_paths than current vertex 
            for following_vertex in vertex_out_neighbors_iterator:
                if highest_parallel_paths_vertex is None or graph.vp['parallel_paths'][following_vertex] < graph.vp['parallel_paths'][highest_parallel_paths_vertex]:
                    highest_parallel_paths_vertex = following_vertex
        
        # group must be contained between those vertices, since this operations has branches.
        vertex_range_start = lowest_parallel_paths_vertex
        vertex_range_start_id = graph.vp['op_id'][vertex_range_start]
        vertex_range_end = highest_parallel_paths_vertex
        vertex_range_end_id = graph.vp['op_id'][vertex_range_end]


        # Get the starting nodes of each branch according to the group description
        print('[WF] Identifying the open front...') if debug else None
        branch_starters = []
        for branch in self.branches:
            branch_starters.extend(branch.getStartingOperations())
        print('[WF] Branch starters:', branch_starters) if debug else None
        # Locate the first nodes that match the branch starters in the nodes not already included.
        potential_starting_vertices = []
        pending_vertices_op_types = [graph.vp['op_type'][pending_vertex].strip().lower().replace("\"", "") for pending_vertex in vertices_not_already_included]
        print('[WF] pending_vertices_op_types:', pending_vertices_op_types) if debug else None
        branch_starters_copy = branch_starters[:]
        for vertex in vertices_not_already_included:
            if graph.vp['op_type'][vertex] is not None:
                node_type = graph.vp['op_type'][vertex].strip().lower().replace("\"", "")
                if node_type in branch_starters_copy and graph.vp['op_id'][vertex] >= vertex_range_start_id and graph.vp['op_id'][vertex] <= vertex_range_end_id:
                    potential_starting_vertices.append(vertex)
                    branch_starters_copy.remove(node_type)
        print('[WF] Potential starting nodes:', [graph.vp['name'][vertex] for vertex in potential_starting_vertices]) if debug else None
        

        # Expand common node following nodes in width-first search until all branch starters are in the front.
        first_common_vertex = vertex_range_start
        print('[WF] Common node:', graph.vp['name'][first_common_vertex]) if debug else None
        forward_facing_front = [first_common_vertex]
        
        branch_starters_in_front = False
        branch_starters_copy = branch_starters[:]
        while not branch_starters_in_front:
            next_step_vertices = []
            # Explore all nodes in the forward front one time, replacing front with all nodes after the current front
            for vertex in forward_facing_front:
                if vertex not in vertices_not_already_included:
                    # Node has already been processed, move to nodes further ahead on the graph
                    print('[WF] Node', graph.vp['name'][vertex], 'has been already processed, advancing to following nodes.' ) if debug else None
                    valid_neighbors = [neighbor for neighbor in vertex.out_neighbors() if graph.vp['op_id'][neighbor] > vertex_range_start_id and graph.vp['op_id'][neighbor]<vertex_range_end_id]
                    next_step_vertices.extend(valid_neighbors)
                    print('[WF] Adding vertices to next_step_nodes:', [graph.vp['name'][neighbor] for neighbor in valid_neighbors]) if debug else None
                else:
                    # Node is still pending to be explored. Check if it is a branch starter
                    if graph.vp['op_type'][vertex].strip().lower().replace("\"", "") in branch_starters_copy and graph.vp['op_id'][vertex] > vertex_range_start_id and graph.vp['op_id'][vertex]<vertex_range_end_id:
                        print('[WF] Adding', graph.vp['name'][vertex], 'to next_step_vertices.' ) if debug else None
                        # Node can start a branch, add to next_step_nodes so it persist to next step without advancing.
                        next_step_vertices.append(vertex)
                        # Remove from branch_starters_copy since we have already matched this branch
                        branch_starters_copy.remove(graph.vp['op_type'][vertex].strip().lower().replace("\"", ""))
                    else:
                        # Node cannot start branches. Move past it
                        valid_neighbors = [neighbor for neighbor in vertex.out_neighbors() if graph.vp['op_id'][neighbor] > vertex_range_start_id and graph.vp['op_id'][neighbor]<vertex_range_end_id]
                        print('[WF] Vertex', graph.vp['name'][vertex], 'cannot start any branch, advancing to vertices connected to this one:', [graph.vp['name'][neighbor] for neighbor in valid_neighbors]) if debug else None
                        next_step_vertices.extend(valid_neighbors)
            # For vertex in forward_facing_front end.

            # Replace current front with next one
            print('[WF] next_step_vertices:', [graph.vp['name'][vertex] for vertex in  next_step_vertices]) if debug else None
            forward_facing_front = next_step_vertices
            candidate_front = []
            # Check if all branches have nodes available to start
            if len(branch_starters) <= len(forward_facing_front):
                # Sizes are compatible, there are equal or more nodes in the front than branch starters are required.
                print('[WF] Sizes compatible, checking all branches can be started with this front.', ) if debug else None
                branch_starters_in_front = True
                # Check each Branch
                available_front_vertex_types = [graph.vp['op_type'][front_vertex].strip().lower().replace("\"", "") for front_vertex in forward_facing_front]
                print('[WF] \tavailable_front_node_types:', available_front_vertex_types) if debug else None
                for branch in self.branches:
                    print('[WF] \tTesting branch:', branch) if debug else None
                    # All branch starter nodes must be present
                    for branch_start_op_type in branch.getStartingOperations():
                        print('[WF]\tChecking starter node:', branch_start_op_type) if debug else None
                        if branch_start_op_type not in available_front_vertex_types:
                            print('[WF] \t\t\tNode not found in available_front_node_types.') if debug else None
                            # No node of the correct type remaining in the forward_facing_front nodes 
                            branch_starters_in_front = False
                            break # Stop for branch_start_op_type in branch.getStartingOperations() loop, at least one branch does not have correct op type in front vertices.
                        else:
                            # Node present, remove from available_front_node_types to not use it again for another branch.
                            print('[WF] \t\t\tVertex found in available_front_vertex_types, removing from remaining available_front_vertex_types.') if debug else None
                            vertex_index = available_front_vertex_types.index(branch_start_op_type)
                            available_front_vertex_types.remove(branch_start_op_type)# Remove node from available_front_node_types
                            candidate_front.append(forward_facing_front[vertex_index])# Add node to candidate front
                            forward_facing_front.pop(vertex_index)# Remove node from front
                # branch_starters_in_front will be False if any branch failed to find a matching in the forward_facing_front, ready for check in the next iteration of the while loop.
                if branch_starters_in_front:
                    forward_facing_front = candidate_front # Replace forward_facing_front so that the front only contains nodes that start branches. Only do it if all branches found matches.
            else:
                if len(forward_facing_front)==0:
                    # No nodes in the front, stop while loop and return empty front
                    print('[WF] Stopping wide front search due to no nodes left in the front.') if debug else None
                    return forward_facing_front
        # Remove nodes that are not branch starters from the front
        print('[WF] Removing nodes not present in branch starters.') if debug else None
        non_branch_starter_vertices = []
        for vertex in forward_facing_front:
            if graph.vp['op_type'][vertex].strip().lower().replace("\"", "") not in branch_starters:
                # Node type is not of the type required for branch starters, is not relevant for the group front
                print('[WF]', graph.vp['op_type'][vertex].strip().lower().replace("\"", ""), 'not in', branch_starters) if debug else None
                non_branch_starter_vertices.append(vertex)
            if vertex not in vertices_not_already_included:
                # Node is already processed, cannot be used as group front
                non_branch_starter_vertices.append(vertex)
        for vertex in non_branch_starter_vertices:
            print('[WF] Removing node from resulting front:', graph.vp['name'][vertex]) if debug else None
            forward_facing_front.remove(vertex)

        # Select and return branch starter nodes from the front.
        print('[WF] Resulting front:', [graph.vp['name'][vertex] for vertex in forward_facing_front]) if debug else None
        return forward_facing_front

    def __init__(self, branches: list[operationDescription]):
        self.branches=branches

    def getStartingOperations(self)->list[str]:
        starting_operations = []
        for branch in self.branches:
            starting_operations.extend(branch.getStartingOperations())
        #return [branch.getStartingOperations() for branch in self.branches]
        return starting_operations

    @classmethod
    def buildFromXMLNode(cls, node, operationsLibrary):
        # Return a list of operations, using the abstract class method to correctly instance each subelement
        return branchedStartOperations([operationDescription.buildFromXMLNode(subnode, operationsLibrary) for subnode in list(node)])

    def __str__(self):
        result = 'Branched start {\n'
        for branch in self.branches:
            result += branch.__str__()+'\n--------\n'
        return result[:-10]+'\n--}'

    def isOperationNextInGraph(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> bool:
        print('[BSc] Checking for branched start: Description:\n\t\t', self.__str__()) if debug else None


        vertices_not_already_included_copy = vertices_not_already_included[:] # Copy list to void modifications to spill outside this method
        # Count non empty branches
        non_empty_branches = len([branch for branch in self.branches if not isinstance(branch, emptyBranch)])

        print('\n[BSc] Getting open front:', [graph.vp['name'][vertex] for vertex in next_vertices_in_graph]) if debug else None
        wide_vertex_front = self.__getOpenFront__(graph, next_vertices_in_graph, vertices_not_already_included_copy, debug=debug)
        print('[BSc] Wide node front:', [graph.vp['name'][vertex] for vertex in wide_vertex_front]) if debug else None

        # If we have more branches that nodes in the wide front (not counting empty branches), it is not this group.
        if non_empty_branches > len(wide_vertex_front):
            print('[BSc] Cannot locate enough nodes for all of the branches:', non_empty_branches, '<', len(wide_vertex_front)) if debug else None
            return False

        # For every branch, check if any of the nodes in the wide front matches the branch.
        included_vertices_in_branches = []
        for branch in self.branches:
            branch_matched = False
            for vertex in wide_vertex_front:
                print('[BSc] Checking branch', branch, 'against node', graph.vp['name'][vertex]) if debug else None
                print('[BSc] Checking isOperationNextInGraph...') if debug else None
                if branch.isOperationNextInGraph(graph, [vertex], vertices_not_already_included_copy, operationsLibrary, debug=debug):
                    # Match found. Mark node for removal from wide_node_front. 
                    print('[BSc] Match for this branch found:', branch) if debug else None
                    branch_matched = True
                    # advance front to retrieve which nodes are included in this branch to remove them from the vertices_not_already_included
                    print('[BSc] calling advanceOperationFront...') if debug else None
                    next_vertices, new_included_vertices = branch.advanceOperationFront(graph, [vertex], vertices_not_already_included_copy, operationsLibrary, debug=debug)     
                    included_vertices_in_branches.extend(new_included_vertices)     
                    break
            # For node in wide_node_front: loop end.
            # If branch_matched is True, some node starts this branch, remove that node from wide front, remove included nodes in this branch from vertices_not_already_included_copy and check the remaining branches
            if branch_matched:
                print('[BSc]\t\tBranch matched, removing nodes included in the branch from wide front and vertices_not_already_included_copy') if debug else None
                for vertex in included_vertices_in_branches:
                    if vertex in vertices_not_already_included_copy:
                        vertices_not_already_included_copy.remove(vertex)
                        print('[BSc]\t\tRemoving node', graph.vp['name'][vertex], 'from vertices_not_already_included_copy for remaining branches') if debug else None
                    if vertex in wide_vertex_front:
                        # If nested operations are also branched start, they can cover multiple nodes from the front.
                        wide_vertex_front.remove(vertex)
                        print('[BSc]\t\tRemoving node', graph.vp['name'][vertex], 'from wide front. Remaining:', [graph.vp['name'][vertex] for vertex in wide_vertex_front]) if debug else None
            else:
                # No node in wide front matches this branch, it is not this group.
                print('[BSc] No node in wide front to match pending branch') if debug else None
                return False
        # If this point is reached, all branches found matching nodes.
        print('[BSc] ALL BRANCHES FOUND MATCHING NODES.') if debug else None
        return True

    def advanceOperationFront(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> Tuple[list[gt.Vertex], list[gt.Vertex]]:
        print('[BSa] ADVANCING BRANCHED START OPERATIONS...') if debug else None

        vertices_not_already_included_copy = vertices_not_already_included[:] # Copy list to void modifications to spill outside this method
        # Locate non included nodes with a edge to already included nodes. This is similar to getting the full width node front.
        print('\n[BSa] Getting open front:', [graph.vp['name'][vertex] for vertex in next_vertices_in_graph]) if debug else None
        wide_vertex_front = self.__getOpenFront__(graph, next_vertices_in_graph, vertices_not_already_included, debug=debug)
        included_vertices_in_branches = []
        next_vertices_after_branches = []
        
        # For every branch, check if any of the nodes in the wide front matches the branch.
        for branch in self.branches:
            branch_matched = False
            for vertex in wide_vertex_front:
                print('[BSa] Checking isOperationNextInGraph...') if debug else None
                if branch.isOperationNextInGraph(graph, [vertex], vertices_not_already_included_copy, operationsLibrary, debug=debug):
                    # Match found. Mark node for removal from wide_node_front. 
                    branch_matched = True
                    # advance front to retrieve which nodes are included in this branch to remove them from the vertices_not_already_included
                    print('[BSa] calling advanceOperationFront...') if debug else None
                    next_vertices, new_included_vertices = branch.advanceOperationFront(graph, [vertex], vertices_not_already_included_copy, operationsLibrary, debug=debug)
                    print('[BSa]\t\tBranch matched:', branch, '\n[BSa]\t\tNext nodes:', [graph.vp['name'][vertex] for vertex in next_vertices], '\n[BSa]\t\tNew_included_nodes:', [graph.vp['name'][vertex] for vertex in new_included_vertices]) if debug else None
                    included_vertices_in_branches.extend(new_included_vertices)
                    for vertex in next_vertices:
                        if vertex not in next_vertices_after_branches:
                            next_vertices_after_branches.append(vertex)    
                    break
            # For node in wide_node_front: loop end.
            # If branch_matched is True, some node starts this branch, remove that node from wide front, remove included nodes in this branch from vertices_not_already_included_copy and check the remaining branches
            if branch_matched:
                print('[BSa]\t\tBranch matched, removing nodes included in the branch from wide front and vertices_not_already_included_copy:\n[BSa] wide_node_front:', 
                        [graph.vp['name'][vertex] for vertex in wide_vertex_front], '\n[BSa]vertices_not_already_included_copy:', [graph.vp['name'][vertex] for vertex in vertices_not_already_included_copy]) if debug else None
                for vertex in included_vertices_in_branches:
                    if vertex in vertices_not_already_included_copy:
                        vertices_not_already_included_copy.remove(vertex)
                        print('[BSa]\t\tRemoving node', graph.vp['name'][vertex], 'from vertices_not_already_included_copy for remaining branches') if debug else None
                    if vertex in wide_vertex_front:
                        # If nested operations are also branched start, they can cover multiple nodes from the front.
                        wide_vertex_front.remove(vertex)
                        print('[BSa]\t\tRemoving node', graph.vp['name'][vertex], 'from wide front. Remaining:', [graph.vp['name'][vertex] for vertex in wide_vertex_front]) if debug else None
            else:
                # No mtach found for branch, raise exception, this method should be called only if isOperationNextInGraph returned true.
                raise RuntimeError('No matching nodes for branched start branch.')
        # If this point is reached, all branches found matching nodes.
        print('[BSa] ADVANCED BRANCHED START OPERATIONS TO NEXT NODES:', [graph.vp['name'][vertex] for vertex in next_vertices_after_branches]) if debug else None
        return next_vertices_after_branches, included_vertices_in_branches




class branchingOperations(operationDescription):

    def __init__(self, branches: list[operationDescription]):
        self.branches=branches

    def getStartingOperations(self)->list[str]:
        starting_operations = []
        for branch in self.branches:
            starting_operations.extend(branch.getStartingOperations())
        return starting_operations

    @classmethod
    def buildFromXMLNode(cls, node, operationsLibrary):
        # Return a list of operations, using the abstract class method to correctly instance each subelement
        return branchingOperations([operationDescription.buildFromXMLNode(subnode, operationsLibrary) for subnode in list(node)])

    def stringSummary(self, tablevel: int) -> str:
        summary = ('\t'*tablevel)+'Branching:\n'
        for branch in self.branches:
            if branch is not None:
                summary += branch.stringSummary(tablevel+1)+'\n'
            else:
                summary += ('\t'*(tablevel+1))+'Empty branch.'+'\n'
        return summary
    
    def __str__(self):
        result = 'Branching ['
        for branch in self.branches:
            result += branch.__str__()+' | '
        return result[:-3]+']'

    def isOperationNextInGraph(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> bool:
        print('[BOc] Checking for branching part...') if debug else None

        # Count non empty branches
        non_empty_branches = [branch for branch in self.branches if not isinstance(branch, emptyBranch)]

        local_next_nodes = next_vertices_in_graph[:]# Copy list to keep track of nodes already checked for branches.

        # Check every branch is valid starting on some of the next nodes. 
        valid_operation = True # Starts as true, if any branch is not valid, changes to false.
        #for index, branch in enumerate(self.branches): # Version matching each branch with each node in next nodes 
        for branch in non_empty_branches: # Version where any next node can match any non empty branch.
            # Check is any next node is valid for this branch
            print('[BOc]\tLooking for match in branch', branch) if debug else None
            any_branch_valid = False

            # Version matching each branch with each node in next nodes.
            #node = next_vertices_in_graph[index]
            #if node.get('op_id') in [local_node.get('op_id') for local_node in local_next_nodes] and branch.isOperationNextInGraph(graph, [node], vertices_not_already_included, operationsLibrary, debug=debug):
            #        any_branch_valid = True
            #        #local_next_nodes.remove(node) # remove from next_nodes list to avoid modifying the original list which we are iterating.
            #        print('\t\t\tMatched branch') if debug else None

            # Version where any next node can match any non empty branch.
            for vertex in next_vertices_in_graph:
                # Check that node has not already matched other branch, and that is matches this branch.
                if vertex in local_next_nodes and branch.isOperationNextInGraph(graph, [vertex], vertices_not_already_included, operationsLibrary, debug=debug):
                    any_branch_valid = True
                    local_next_nodes.remove(vertex) # remove from next_nodes list to avoid modifying the original list which we are iterating.
                    print('[BOc]\t\tMatched branch') if debug else None
                    break
            # Stop checks if the branch checked all nodes and none is valid for this branch, therefore, not all branches can start and they
            if not any_branch_valid:
                valid_operation = False
                print('[BOc]\t\tNo vertex is valid for this branch') if debug else None
                break
        # For loop stopped/finished. Return whether or not all branches were valid
        print('[BOc] CHECK FOR BRANCHING OPERATIONS:', valid_operation) if debug else None
        return valid_operation



    def advanceOperationFront(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> Tuple[list[gt.Vertex], list[gt.Vertex]]:

        local_next_vertices = next_vertices_in_graph[:]# Copy list to keep track of nodes already used for branches.
        new_next_vertices: list[gt.Vertex] = []
        included_vertices: list[gt.Vertex] = []
        print('[BOa]\tADVANCING BRANCHING OPERATIONS') if debug else None
        print('[BOa]\tlocal_next_nodes:', [graph.vp['name'][vertex] for vertex in local_next_vertices]) if debug else None
        # Advance the front of every branch
        non_empty_branches = [branch for branch in self.branches if not isinstance(branch, emptyBranch)]
        for branch in non_empty_branches:
            # Check which node in next_vertices_in_graph is the one that corresponds to this branch.
            for vertex in next_vertices_in_graph:
                if vertex in local_next_vertices and branch.isOperationNextInGraph(graph, [vertex], vertices_not_already_included, operationsLibrary, debug=debug):
                    print('[BOa]\tValid branch found, advancing nodes on the branch:', branch) if debug else None
                    branch_next_vertices, branch_included_vertices = branch.advanceOperationFront(graph, [vertex], vertices_not_already_included, operationsLibrary, debug=debug)
                    included_vertices.extend(branch_included_vertices) # Should be all nodes not included in other branches. No need to check for duplicates.

                    if vertex in branch_included_vertices: # Remove start point of the branch if the branch includes it (empty branches don't include any node)
                        local_next_vertices.remove(vertex) # remove from new_nodes list to avoid modifying the original list which we are iterating.
                    # Add branch_next_nodes nodes to new_next_nodes if they are not already in it
                    for next_vertex in branch_next_vertices:
                        if next_vertex not in new_next_vertices:
                            new_next_vertices.append(next_vertex)
                    print('[BOa]\tProcessed vertices:', [graph.vp['name'][vertex] for vertex in branch_included_vertices], ' branch_next_vertices:', [graph.vp['name'][vertex] for vertex in branch_next_vertices], 
                        'new_next_vertices:', [graph.vp['name'][vertex] for vertex in new_next_vertices]) if debug else None
                    break # Skip to next branch.
        
        # All branches advanced, return new next nodes and included nodes
        print('[BOa] ADVANCED BRANCHING OPERATIONS TO NEXT NODES:', [graph.vp['name'][vertex] for vertex in new_next_vertices]) if debug else None
        return new_next_vertices, included_vertices


class miscellaneousOperationsSet(operationDescription):

    def __init__(self, possibleOperations: list[miscelaneousOperationType]):
        self.possibleOperations = possibleOperations

    def getStartingOperations(self)->list[str]:
        return [operation.getName() for operation in self.possibleOperations]

    @classmethod
    def buildFromXMLNode(cls, node, operationsLibrary):
        """
        Build an opration description from xml node. Creates an operation description capable os matching any amount of miscelaneousOperationTypes
        Args:
            node (_type_): _description_
            operationsLibrary (_type_): _description_

        Returns:
            _type_: _description_
        """
        return miscellaneousOperationsSet(operationsLibrary.getMiscellaneousOperations())

    def stringSummary(self, tablevel: int) -> str:
        return ('\t'*tablevel)+'Miscellaneous nodes sequence'

    def __str__(self):
        return 'Miscellaneous operations'

    def isOperationNextInGraph(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> bool:
        print('[MOc]Checking miscellaneous set of operations...') if debug else None

        # Handles branching as long as all branches are of misc operations
        misc_operations_names = [op.getName().strip().lower().replace("\"", "") for op in operationsLibrary.getMiscellaneousOperations()]
        print('[MOc]\tSupported miscellaneous set of operations:', misc_operations_names) if debug else None

        # Check each next node to see if any is a misc node.
        for next_vertex in next_vertices_in_graph:
            print('[MOc]\tTesting vertex', next_vertex.get('op_type'), 'in misc ops' ) if debug else None
            if graph.vp['op_type'][next_vertex].strip().lower().replace("\"", "") in misc_operations_names:
                # At least one of the next nodes is a miscellaneous node, return True
                print('[MOc]\tAt least one misc op' ) if debug else None
                return True
        # None of the next nodes is considered a misc node, therefore there is no misc node group.
        return False


    
    def advanceOperationFront(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> Tuple[list[gt.Vertex], list[gt.Vertex]]:

        local_next_vertices = next_vertices_in_graph[:]
        misc_operations_names = [op.getName().strip().lower().replace("\"", "") for op in operationsLibrary.getMiscellaneousOperations()]
        included_vertices: list[gt.Vertex] = []

        some_misc_operations = True
        while some_misc_operations:
            # Advance one operation on each next node
            new_next_vertices: list[gt.Vertex] = []
            
            for next_vertex in local_next_vertices:
                if graph.vp['op_type'][next_vertex].strip().lower().replace("\"", "") in misc_operations_names:
                    # Operation can be considered misc. Move to next op in graph.
                    included_vertices.append(next_vertex)
                    # Add outgoing neighbors of the vertex to new_next_vertices
                    new_next_vertices.extend(next_vertex.out_neighbors())

                else:
                    # Not a misc operation. Add to new_next_nodes to keep it
                    new_next_vertices.append(next_vertex)

            # All nodes in local_next_nodes processed. Check if any misc operation remains in new_next_nodes
            some_misc_operations = False
            for new_next_vertex in new_next_vertices:
                if graph.vp['op_type'][new_next_vertex].strip().lower().replace("\"", "") in misc_operations_names:
                    some_misc_operations = True
            # Replace local_next_nodes with new_next_nodes
            local_next_vertices = new_next_vertices

        # While ended, no more misc operations are left in the next nodes.
        return local_next_vertices, included_vertices


class crossBranchJump(operationDescription):
    """
        Represent an empty branch that connects to another branch in the same operation group.
    Args:
        operationDescription (_type_): _description_
    """

    def getStartingOperations(self)->list[str]:
        return []
    
    @classmethod
    def buildFromXMLNode(cls, node, operationsLibrary):
        return crossBranchJump()

    def stringSummary(self, tablevel: int) -> str:
        return ('\t'*tablevel)+'Cross-Branch jump'
    
    def __str__(self):
        return 'Cross-Branch jump'
    
    def isOperationNextInGraph(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> bool:
        """
        Cross-Branch jumps are always present since they have no nodes that can be not present.
        Args:
            graph (gt.Graph): _description_
            non_grouped_nodes_remaining (list[gt.Vertex]): _description_
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            bool: True.
        """ 
        print('[CJc]\tCross-Branch jump to:', [graph.vp['name'][vertex] for vertex in next_vertices_in_graph]) if debug else None
        return True

    def advanceOperationFront(
        self,
        graph: gt.Graph,
        next_vertices_in_graph: list[gt.Vertex],
        vertices_not_already_included: list[gt.Vertex],
        operationsLibrary,
        debug: bool = False,
    )-> Tuple[list[gt.Vertex], list[gt.Vertex]]:
        """
            Returns an empty list to eliminate one node from the next nodes list.
        Args:
            graph (gt.Graph): _description_
            next_vertices_in_graph (list[gt.Vertex]): _description_
            vertices_not_already_included (list[gt.Vertex]): _description_
            operationsLibrary (_type_): _description_
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[list[gt.Vertex], list[gt.Vertex]]: Two empty lists. This operation terminates a branch without duplicating next nodes and does not include any graph node.
        """
        print('[CJa]\tADVANCED Cross-Branch jump.') if debug else None
        return [], []
    