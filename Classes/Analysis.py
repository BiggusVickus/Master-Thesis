import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
from Classes.Math import determine_type_of_variable
np.seterr(divide='ignore')
import warnings
warnings.filterwarnings("ignore", message="The following arguments have no effect for a chosen solver: `min_step`.")

class Analysis():
    def __init__(self, graph_location):
        """Given a graph file, this class will read in the graph file and store the graph as an attribute. It will also store the location of the graph file as an attribute. The class contains methods to extract the data from nodes and return it as a vector. It also contains methods to initialize matrices and vectors from the graph file and store the parameters of that data in the matrix to do vector-matrix calculations in the ODE system. The class also contains a method to solve the ODE system given the ODE system function, initial conditions, and parameters, and has a method to check for cutoff values to set to 0. Given an evironment node, it will store the parameters of the environment as attributes of the class, like temeprature, pH, time step, and simulation length, or any other given parameters.

        The user needs to provide their own implementation of the ODE system function, and the class will take care of the rest of the calculations and data extraction from the graph file. The user needs to ensure that the flattened vector is unflattened to the associated arrays and matrices to do the calculations, and then reflattened to return the derivative of the system. The user needs to provide the extra parameters to use when calling solve_system() for use in the method. The user needs to unpack the *args values correctly to use those parameters. The user can optionally check for cutoff values by calling check_cutoff() to set small values to 0. The user needs to repack the ODE data into a single vector to then be calculated. 
        To do this, the user can implement a dummy class that inherits from this class, and then implement the ODE system function in that class. The user can also then override the default flatten and unflatten methods to do the calculations.
        The user can also provide any extra parameters to the solve_ivp function, like the method to use, the number of steps, or any other parameters that the solve_ivp function can take in. 

        Args:
            graph_location (str): location of the graph file to be read in
        """
        self.graph_location = graph_location # graph file location
        self.graph = nx.read_gexf(graph_location) # read in the graph file
        self.settings = {} # settings for the class, can be used to set the parameters for the ODE system, like the time step, simulation length, and cutoff value
        self.min_step = 0.01 # default min step, backup in case the user does not set it
        self.max_step = 0.1 # default max step, backup in case the user does not set it
        self.simulation_length = 24 # default simulation length, backup in case the user does not set it
        self.cutoff_value = 0.000001 # default cutoff value, backup in case the user does not set it
        self.solver_type = 'RK45' # default solver type, backup in case the user does not set it
        self.dense_output = False # default dense output, backup in case the user does not set it
        self.t_eval_option = False # default t_eval option, backup in case the user does not set it
        self.t_start = 0 # default t_start, backup in case the user does not set it
        self.t_eval_steps = 200 # default t_eval steps, backup in case the user does not set it
    
    def odesystem(self, t, y, *args):
        """The user must provide their own implementation of the ODE system function. The user can program the function in any way they see fit, but it must take in the time, the current state of the system, and any parameters needed to calculate the ODE system, and return the derivative of the system at that time. The function must be in the form of f(t, y, *args) -> np.array. The user can implement how they see fit, with for loops or with matrix-vector calculations, but they need to make sure that they unpack the y0_flattened vector into the correct matrices and vectors to do the calculations. The function must return the derivative of the system at that time in a reflattened vector.

        The easiest way to do this is to implement the function in a dummy class that inherits from this class, and then implement the ODE system function in that class. The user can also then use and/or override the defualt flatten and unflatten methods to do the calculations to help fit the dimensions of the matrices and vectors. The user can also provide any extra parameters to the solve_ivp function, like the method to use, the number of steps, or any other parameters that the solve_ivp function can take in.

        Args:
            t (float): time at iteration
            y (array): the current state of the system, unflattened array of data wanting to be calculated
            args (any): can be any parameters or varaibles used to calculate (or help assist in calcualting) the ODE system. The user can pass in any number of parameters or variables to be used in the ODE system calculations. These would commonly be a graph object for checking if edges exist between nodes, parameter vectors and matrices, etc. The user must make sure that the parameters are unpacked correctly from the *args in the ODE_system_function to do the calculations using the parameters.
        """
        pass
    
    def get_nodes_of_type(self, node_type:str):
        """Given a particular node type, for example of phage P, bacteria B, or resource R, this method will return a list of all the node names of that type in the graph. It will also store the names of the nodes as an attribute of the class. If the node type is an environment or setting node, it will store the parameters of the environment as attributes of the class using add_item_to_class_attribute(). The user can then access the data as an attribute of the class by calling YourClassName.name, assuming that YourClassName extends this Analysis class. You can also directly call add_item_to_class_attribute() to add any extra parameters to the class that are not part of the graph, that you might want to later use for whatever reason. 

        Args:
            node_type (str): The type of node to be extracted from the graph, will usually be either 'P', 'B', 'R', or 'E'. The user can also add their own node types, but they need to make sure that the node type is in the graph file.

        Returns:
            list: List of all the node names of the given node type
        """
        # create the list of nodes of the given type
        list_data =  [n for n in self.graph.nodes if self.graph.nodes[n]['node_type'] == node_type]
        # if the node type is an environment node, get the data from the node and store it as an attribute of the class
        if node_type == "E":
            self.environment_data = self.turn_string_to_dictionary(self.graph.nodes[list_data[0]]['data'])
            for d, v in self.turn_string_to_dictionary(self.graph.nodes[list_data[0]]['data']).items():
                setattr(self, d, float(v))
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
        # new dictionary to be returned
        dictionary = {}
        # split the string by the new line character, and then split each entry by the ':' character
        for row in string.split('\n'):
            row = row.split(':')
            # ensure there is only 2 entries in the row, and then add the key and value to the dictionary
            if len(row) == 2:
                dictionary[row[0]] = row[1]
        return dictionary

    def update_environment_data(self, dictionary:dict) -> None:
        """Adds the environment data to the class as an attribute

        Args:
            dictionary (str):Dictionary of environment data to be added to the class, of the form {'key':'value', ...}, where the attribute name is key, and the attirbute value is value. 
        """
        # loop through the dictionary and set the attributes of the class to the values in the dictionary
        for key, value in dictionary.items():
            new_value = determine_type_of_variable(value)
            setattr(self, key, new_value)
            dictionary[key] = new_value
        return dictionary        
    
    def initialize_new_matrix(self, rows:int, columns:int):
        """Initializes a new matrix of zeros with the given number of rows and columns. The matrix is initialized as a numpy array.

        Args:
            rows (int): Number of rows for the matrix
            columns (int): Number of columns for the matrix

        Returns:
            np.array: 2D np array of size rows x columns
        """
        return np.zeros((int(rows), int(columns)))
    
    def initialize_new_vector(self, rows):
        """Initializes a new vector of zeros with the given number of rows. The vector is initialized as a numpy array.
        
        Args:
            rows (int): Number of rows for the vector
        
        Returns:
            np.array: 1D np array of size rows
        """
        return np.zeros(int(rows))
    
    def initialize_new_parameter_from_edges(self, attribute_name:str, node_list1:list, node_list2:list, data_type = float) -> np.array:
        """Returns a new matrix consisting of the data from the edges between the nodes in node_list1 and node_list2 for a listed attribute name. The data is stored in the matrix as the data_type given, in case the data is not a float. The data is extracted from the attribute_name given.

        Args:
            attribute_name (str): The attribute name to be extracted from the edges
            node_list1 (list): List of nodes to be used as the rows of the matrix
            node_list2 (list): List of nodes to be used as the columns of the matrix
            data_type (_type_, optional): Optional, the datatype to convert from string to. Defaults to float.

        Returns:
            np.array: _description_
        """
        # initialize the matrix with the number of rows and columns given by the length of the node lists
        matrix = np.zeros((len(node_list1), len(node_list2)))
        # loop through the nodes in node_list1 and node_list2, and check if there is an edge between them. If there is, get the data from the edge and store it in the matrix
        for node1 in node_list1:
            for node2 in node_list2:
                if self.graph.has_edge(node1, node2):
                    data = self.turn_string_to_dictionary(self.graph[node1][node2]['data'])
                    matrix[node_list1.index(node1), node_list2.index(node2)] = data_type(data[attribute_name])
        return matrix
    
    def initialize_new_parameter_from_node(self, attribute_name:str, node_list1:list, data_type = float) -> np.array:
        """Initializes a new vector with the number of rows given by th elength of node_list1. The vector is initialized as a numpy array. The data is extracted from the attribute_name given from the nodes given in node_list1. The data is stored in the vector as the data_type given, in case the data is not a float. 

        Args:
            attribute_name (str): name of the attribute to be extracted from the nodes
            node_list1 (list): List of nodes to be used as the rows of the vector
            data_type (type, optional): The datatype needed/wanted. Can be int for example. Defaults to float.

        Returns:
            np.array: np array (vector) of size length of node_list1 with the attribute data extracted from the nodes given by attribute_name
        """
        # create vector of length of node_list1
        vector = np.zeros(len(node_list1))
        for node1 in node_list1: # loop through the nodes in node_list1
            data = self.turn_string_to_dictionary(self.graph.nodes[node1]['data']) # get the data from the node
            # check to see if the data is in the vector at the index of the node in node_list1
            if attribute_name in data:
                vector[node_list1.index(node1)] = data_type(data[attribute_name])
        return vector
    
    def flatten_lists_and_matrices(self, *initial_populations:np.array) -> np.array:
        """Flattens any number of vectors and/or matrices into a single vector in order

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
        If length is an int, it will unflatten the matrix into a matrix of the given length.
        If length is a list, it will unflatten the matrix into submatrices of the lengths specified in the list.

        Returns:
            np.array: The unflattened matrix
        """
        if isinstance(length, int): # if length is an int, unflatten the matrix into a matrix of the given length, at intervals of every length
            return [[vector[i:i+length] for i in range(0, len(vector), length)]]
        elif isinstance(length, list): # if length is a list, unflatten the matrix into submatrices of the lengths specified in the list
            result = [] # initialize the result list and index
            index = 0
            for l in length: # loop through the lengths in the list
                if type(l) == int: # if the length is an int, unflatten the matrix into a matrix of the given length
                    result.append(vector[index:index+l])
                    index += l
                elif type(l) == tuple: # if the length is a tuple, unflatten the matrix into a matrix of the given length, reshaped by the tuple
                    result.append(np.array(vector[index:index+int(np.prod(l))]).reshape(l))
                    index += int(np.prod(l))
            return result
        else:
            raise ValueError("Length must be an int or a list of ints")

    def solve_system(self, ODE_system_function, y0_flattened:np.array, *ODE_system_parameters, t_start = None, t_end = None, **extra_parameters) -> np.array:
        """Solves the system of ODEs given the ODE system function, initial conditions, and parameters

        Args:
            ODE_system_function (function): User provided implementation of a function that can be used to calculate the ODE system. The user can program the function in any way they see fit, but it must take in the time, the current state of the system, and any parameters needed to calculate the ODE system, and return the derivative of the system at that time. The function must be in the form of f(t, y, *args) -> np.array. The user can implement how they see fit, with for loops or with matrix-vector calculations, but they need to make sure that they unpack the y0_flattened vector into the correct matrices and vectors to do the calculations. The function must return the derivative of the system at that time in a reflattened vector.
            y0_flattened (np.array): The initial conditions of the system, flattened into a single vector. The user must make sure that the initial conditions are unpacked correctly in the ODE_system_function to do the calculations. Then at the end, need to be repacked into a single vector to return the derivative of the system.
            ODE_system_parameters (any): can be any parameters or varaibles used to calculate the ODE system. The user can pass in any number of parameters or variables to be used in the ODE system calculations. The user must make sure that the parameters are unpacked correctly from the *args in the ODE_system_function to do the calculations using the parameters.
            extra_parameters (any): Any extra parameters to be passed into the solve_ivp function. The user can pass in any extra parameters to the solve_ivp function, like the method to use, the number of steps, or any other parameters that the solve_ivp function can take in. The user can pass in any number of extra parameters, and they will be passed into the solve_ivp function.

        Returns:
            np.array: The solution/derivative to the ODE system at time t. The solution is returned as a vector. 
        """
        # check if parameters are in the settings, if not, set it to the time step
        max_step = float(self.max_step) if 'max_step' not in self.settings else float(self.settings['max_step'])
        min_step = float(self.min_step) if 'min_step' not in self.settings else float(self.settings['min_step'])
        simulation_length = float(self.simulation_length) if 'simulation_length' not in self.settings else float(self.settings['simulation_length'])

        if t_start is None:
            t_start = float(self.t_start) if 't_start' not in self.settings else float(self.settings['t_start'])
        if t_end is None:
            t_end = float(t_start + simulation_length)
        t_eval_option = self.t_eval_option if 't_eval_option' not in self.settings else self.settings['t_eval_option']
        t_eval_steps = int(self.t_eval_steps) if 't_eval_steps' not in self.settings else int(self.settings['t_eval_steps'])
        t_eval = np.linspace(t_start, t_end, t_eval_steps) if t_eval_option else None

        solver_type = self.solver_type if 'solver_type' not in self.settings else self.settings['solver_type']
        dense_output = self.dense_output if 'dense_output' not in self.settings else self.settings['dense_output']

        solved = solve_ivp(ODE_system_function, (t_start, t_end), y0_flattened, args=ODE_system_parameters, **extra_parameters, min_step=min_step, max_step=max_step, method=solver_type, dense_output=dense_output, t_eval = t_eval)
        return solved
    
    def add_item_to_class_attribute(self, name:str, data:object) -> None:
        """If you decide to add the 'E' node to 'get_nodes_of_type()', the environment data will be added to the class as an attribute. Otherwise this is useful for adding any extra parameters to the class that are not part of the graph, that you might want to later use for whatever reason.
        
        You can then access the data as an attribute of the class by calling YourClassName.name, assuming that YourClassName extends this Analysis class. This is useful for adding any extra parameters to the class that are not part of the graph, that you might want to later use for whatever reason. You can also add any extra parameters to the class that are not part of the graph, that you might want to later use for whatever reason.

        Args:
            name (str): Name of the attribute to be added
            data (any): Data to be added to the attribute. Datatype can be of any type desired
        """
        setattr(self, name, data)
    
    def check_cutoff(self, flat_array:np.array):
        """Given a flat array, this method will check for any values below the cutoff value and set them to 0. This is for use in the user provided ODE system function to set any values below a certain threshold to 0, just before returning the vector. This is to prevent any numerical errors for values reaching really small values from propagating through the system.

        Args:
            flat_array (np array): The flattened array from the user provided to be checked for cutoff values. Best used at the start of the ODE loop, as the ode solver uses some math behind the scenes to calculate the gradient, affecting the final value plotted. Loops through the array and checks if the value is less than the cutoff value given by in the settings. If it is, it sets the value to 0.

        Returns:
            np array: Returns the flat array with any values at or below the cutoff value set to 0. The array is returned as a numpy array.
        """
        # gets the cutoff value from the settings, if not set, sets it to the default value
        cutoff_value = float(self.cutoff_value) if 'cutoff_value' not in self.settings else float(self.settings['cutoff_value'])
        # loop through the array and check if the value is less than the cutoff value, if yes, set = 0. 
        for index, value in enumerate(flat_array):
            if value <= cutoff_value:
                flat_array[index] = 0
        return flat_array
    
    def prevent_negative_numbers(self, ODE_input:list, ODE_output:list)->np.array:
        """Given a list of ODE inputs and outputs, this method will check for any when the ODE solver is given a value of 0 or less, and the derivative would be negative, it will set the value to 0. This is to prevent any numerical errors for values reaching really small values from propagating through the system, and from preventing values from going negative. 
        Args:
            ODE_input (list): The list of ODE inputs to be checked for negative values
            ODE_output (list): The list of ODE outputs to be checked for negative values

        Returns:
            list: Returns the ODE output list with any negative values set to 0 if the ODE input is less than the cutoff value. The list is returned as a numpy array.
        """
        # loop through the array and check if the value is less than the cutoff value, if yes, set = 0. 
        cutoff_value = float(self.cutoff_value) if 'cutoff_value' not in self.settings else float(self.settings['cutoff_value'])
        for i in range(len(ODE_input)):
            if ODE_input[i] <= cutoff_value and ODE_output[i] < 0:
                ODE_output[i] = 0
        return ODE_output