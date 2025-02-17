import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
np.seterr(divide='ignore')

class Analysis():
    def __init__(self, graph_location):
        """Given a graph file, this class will read in the graph file and store the graph as an attribute. It will also store the location of the graph file as an attribute. The class contains methods to extract the data from nodes and return it as a vector. It also contains methods to initialize matrices and vectors from the graph file and store the parameters of that data in the matrix to do vector-matrix calculations in the ODE system. The class also contains a method to solve the ODE system given the ODE system function, initial conditions, and parameters, an dhas a method to check for cutoff values to set to 0. Given an evironment node, it will store the parameters of the environment as attributes of the class, like temeprature, pH, time step, and simulation length, or any other given parameters.

        The user needs to provide their own implementation of the ODE system function, and the class will take care of the rest of the calculations and data extraction from the graph file. The user needs to ensure that the flattened vector is unflattened to the associated arrays and matrices to do the calculations, and then reflattened to return the derivative of the system. The user needs to provide the extra parameters to use when calling solve_system() for use in the method. The user needs to unpack the *args vlaues correctly to use those parameters. The user can optionally check for cutoff values by calling check_cutoff() to set small values to 0. The user needs to repack the ODE data into a single vector to then be calculated. 
        To do this, the user can implement a dummy class that inherits from this class, and then implement the ODE system function in that class. The user cna also then override the defualt flatten and unflatten methods to do the calculations.
        The user can also provide any extra parameters to the solve_ivp function, like the method to use, the number of steps, or any other parameters that the solve_ivp function can take in. 

        Args:
            graph_location (str): location of the graph file to be read in
        """
        self.graph_location = graph_location 
        self.graph = nx.read_gexf(graph_location)
        self.attribute_additions = []
        self.Simulation_Length = 10
        self.Time_Step = 0.1
    
    def get_nodes_of_type(self, node_type:str):
        """Given a particular node type, for example of phage P, bacteria B, or resource R, this method will return a list of all the node names of that type in the graph. It will also store the names of the nodes as an attribute of the class. If the node type is an environment node, it will store the parameters of the environment as attributes of the class.

        Args:
            node_type (str): The type of node to be extracted from the graph

        Returns:
            list: List of all the node names of the given node type
        """
        list_data =  [n for n in self.graph.nodes if self.graph.nodes[n]['node_type'] == node_type]
        if node_type == "E":
            for d, v in self.turn_string_to_dictionary(self.graph.nodes[list_data[0]]['data']).items():
                setattr(self, d, float(v))
        else:
            self.attribute_additions.append(node_type + '_node_names')
        setattr(self, node_type + '_node_names', list_data)
        return list_data
    
    def load_graph(self):
        """Returns the graph object representing the current state of the graph

        Returns:
            Graph(): Current state of the graph
        """
        return self.graph
    
    def turn_string_to_dictionary(self, string):
        """Enter a string with entries seperated by a '\n', and each entry consisting of key:value, and returns as the format of a dictionary
        Example would be: string = 'key1:value1\nkey2:value2\nkey3:value3' 
        Returned dictionary would be: {'key1':'value1', 'key2':'value2', 'key3':'value3'}

        Args:
            string (str): String with entries seperated by a '\n', and each entry consisting of key:value

        Returns:
            dictionary: Dictionary with keys and values from the string
        """
        dictionary = {}
        for row in string.split('\n'):
            row = row.split(':')
            if len(row) == 2:
                dictionary[row[0]] = row[1]
        return dictionary

    def add_environment_data(self, name:str, data) -> None:
        """Adds the environment data to the class as an attribute

        Args:
            name (str): Name of the attribute to be added
            data (any): Data to be added to the attribute. Datatype can be of any type desired
        """
        setattr(self, name, data)
    
    def initialize_new_matrix(self, rows, columns):
        return np.zeros((int(rows), int(columns)))
    
    def initialize_new_parameter_from_edges(self, node_list1:list, node_list2:list, attribute_name:str, data_type = float) -> np.array:
        """Returns a new matrix consisting of the data from the edges between the nodes in node_list1 and node_list2 for a listed attribute name. The data is stored in the matrix as the data_type given, in case the data is not a float. The data is extracted from the attribute_name given.

        Args:
            node_list1 (list): List of nodes to be used as the rows of the matrix
            node_list2 (list): List of nodes to be used as the columns of the matrix
            attribute_name (str): The attribute name to be extracted from the edges
            data_type (_type_, optional): Optional, the datatype to convert from string to. Defaults to float.

        Returns:
            np.array: _description_
        """
        matrix = np.zeros((len(node_list1), len(node_list2)))
        for node1 in node_list1:
            for node2 in node_list2:
                if self.graph.has_edge(node1, node2):
                    data = self.turn_string_to_dictionary(self.graph[node1][node2]['data'])
                    matrix[node_list1.index(node1), node_list2.index(node2)] = data_type(data[attribute_name])
        return matrix
    
    def initialize_new_parameter_from_node(self, node_list1, attribute_name, data_type = float):
        matrix = np.zeros(len(node_list1))
        for node1 in node_list1:
                data = self.turn_string_to_dictionary(self.graph.nodes[node1]['data'])
                matrix[node_list1.index(node1)] = data_type(data[attribute_name])
        return matrix
    
    def flatten_lists_and_matrices(self, *initial_populations:np.array) -> np.array:
        """Flattens any number of vectors and/or matrices into a single vector

        Returns:
            np.array: Flattened matrix of all the vectors and matrices given
        """
        return np.concatenate([flat.flatten() for flat in initial_populations])

    def unflatten_initial_matrix(self, vector:np.array, length) -> np.array:
        """Given a flattened matrix and a length, this method will unflatten the matrix into a matrix of the given length.
        If length is a list, it will unflatten the matrix into submatrices of the lengths specified in the list.

        Args:
            vector (np.array): The flattened matrix
            length (int or list): The length or list of lengths to unflatten the matrix

        Returns:
            np.array: The unflattened matrix
        """
        if isinstance(length, int):
            return np.array([vector[i:i+length] for i in range(0, len(vector), length)])
        elif isinstance(length, list):
            result = []
            index = 0
            for l in length:
                result.append(vector[index:index+l])
                index += l
            return np.array(result)
        else:
            raise ValueError("Length must be an int or a list of ints")

    def solve_system(self, ODE_system_function, y0_flattened:np.array, *ODE_system_parameters, **extra_parameters) -> np.array:
        """Solves the system of ODEs given the ODE system function, initial conditions, and parameters

        Args:
            ODE_system_function (function): User provided implementation of a function that can be used to calculate the ODE system. The user can program the function in any way they see fit, but it must take in the time, the current state of the system, and any parameters needed to calculate the ODE system, and return the derivative of the system at that time. The function must be in the form of f(t, y, *args) -> np.array. The user can implement how they see fit, with for loops or with matrix-vector calculations, but they need to make sure that they unpack the y0_flattened vector into the correct matrices and vectors to do the calculations. The function must return the derivative of the system at that time in a reflattened vector.
            y0_flattened (np.array): The initial conditions of the system, flattened into a single vector. The user must make sure that the initial conditions are unpacked correctly in the ODE_system_function to do the calculations. Then at the end, need to be repacked into a single vector to return the derivative of the system.
            ODE_system_parameters (any): can be any parameters or varaibles used to calculate the ODE system. The user can pass in any number of parameters or variables to be used in the ODE system calculations. The user must make sure that the parameters are unpacked correctly from the *args in the ODE_system_function to do the calculations using the parameters.
            extra_parameters (any): Any extra parameters to be passed into the solve_ivp function. The user can pass in any extra parameters to the solve_ivp function, like the method to use, the number of steps, or any other parameters that the solve_ivp function can take in. The user can pass in any number of extra parameters, and they will be passed into the solve_ivp function.

        Returns:
            np.array: The solution/derivative to the ODE system at time t. The solution is returned as a vector. 
        """
        return solve_ivp(ODE_system_function, (0, self.Simulation_Length), y0_flattened, args=ODE_system_parameters, **extra_parameters)
    
    def check_cutoff(self, flat_array:np.array, cutoff_value:float = 0.000001):
        """Given a flat array, this method will check for any values below the cutoff value and set them to 0. This is for use in the user provided ODE system function to set any values below a certain threshold to 0, just before returning the vector. This is to prevent any numerical errors for values reaching really small values from propagating through the system.

        Args:
            flat_array (np.array): _description_
            cutoff_value (float, optional): _description_. Defaults to 0.000001.

        Returns:
            _type_: _description_
        """
        if hasattr(self, 'cutoff_value'):
            cutoff_value = self.cutoff_value
        for index, value in enumerate(flat_array):
            if value <= cutoff_value:
                flat_array[index] = 0
        return flat_array