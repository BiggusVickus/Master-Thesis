import numpy as np
from Classes.Analysis import Analysis
from Classes.GraphMakerGUI import GraphMakerGUI
from Classes.Visualizer import Visualizer

# use base class Analysis to create a new class System
class System(Analysis):
    def __init__(self, graph_location):
        """The ODE representation of the Golding Model from 'Using population dynamics to count bacteriophages and their lysogens' from Yuncong Geng, Thu Vu Phuc Nguyen, Ehsan Homaee, and Ido Golding. 
        Uses Classes.Analysis as a base class. 
        Args:
            graph_location (str): location of the graph file
        """
        super().__init__(graph_location)

    def odesystem(self, t, Y, *params):
        """The system of ODEs that represent the Golding Model.
        Args:
            t (float): The time value that the solver is currently at.
            Y (np.array): The initial (for t=0)/current (for t>0) population values of the system. A 1D array of length equal to the number of variables in the system.
        """
        def g(R, v, K):
            """Calculate the growth rate of the bacteria. The growth rate is a function of the concentration of the resource, the growth rate of the bacteria, and the carrying capacity of the bacteria.
            Args:
                R (float): Resource concentration
                v (float): max rate of growth
                K (float): carrying capacity
            Returns:
                float: The growth rate of the bacteria. v*R/(R+K)
            """
            return (R * v) / (R + K)
        # Unpack the parameters. Need to get the graph network, the list of phage/bacteria/resource node names, value of M, the vectors (tau and washin), and the matrices (e, v, K, r, B), and the environment settings
        graph_object, phage_nodes, bacteria_nodes, resource_nodes, M, tau_vector, washin_vector, e_matrix, v_matrix, K_matrix, r_matrix, B_matrix, environment = params
        graph = graph_object.graph
        Y = self.check_cutoff(Y) # check to see if any values are really small. If yes, set to 0
        R, U, I, P = self.unflatten_initial_matrix(Y, [len(resource_nodes), len(bacteria_nodes), (len(bacteria_nodes), M), len(phage_nodes)]) # Turn Y into the shape of the system. 
        new_R = np.zeros_like(R) # create fresh copies of the arrays to be updated. 
        new_U = np.zeros_like(U)
        new_I = np.zeros_like(I)
        new_P = np.zeros_like(P)
        
        #update N vector
        for resource in resource_nodes: # loop over the resource names
            r_index = resource_nodes.index(resource) # get the index of the phage
            washin = washin_vector[r_index]
            sum = 0
            for bacteria in bacteria_nodes: # loop over the bacteria names
                b_index = bacteria_nodes.index(bacteria)
                if graph.has_edge(bacteria, resource): # important to check if the edge between bacteria_b and resource_r exists.
                    e = e_matrix[b_index, r_index] # get the associated values for this interaction
                    v = v_matrix[b_index, r_index]
                    K = K_matrix[b_index, r_index]
                    U_b = U[b_index] # uninfected
                    I_b = np.sum(I[b_index]) # sum of all infected bacteria b agents
                    sum += e * g(R[r_index], v, K) * (U_b + I_b) 
            new_sum = -sum - R[r_index] * environment['washout'] + washin # calculate the new value of the resource.
            if new_sum <= 0 and R[r_index] <= self.cutoff_value: # if the new sum is negative and the resource population is greater than 0, set it to 0
                new_R[r_index] = 0
            else:
                new_R[r_index] = new_sum
        
        # update U vector
        for uninfected_bacteria in bacteria_nodes:
            u_index = bacteria_nodes.index(uninfected_bacteria)
            p_sum = 0
            g_sum = 0
            for resource in resource_nodes: # loop over the resource names
                r_index = resource_nodes.index(resource) # get index of the resource
                if graph.has_edge(uninfected_bacteria, resource):
                    g_sum += g(R[r_index], v_matrix[u_index, r_index], K_matrix[u_index, r_index])
            for phage in phage_nodes: # loop over the phage names
                p_index = phage_nodes.index(phage) # get index of the phage
                if graph.has_edge(phage, uninfected_bacteria): # check if the edge between phage_p and bacteria_b exists.
                    p_sum += r_matrix[p_index, u_index] * P[p_index]
            # update the uninfected bacteria
            new_sum = U[u_index] * g_sum - U[u_index] * p_sum - U[u_index] * environment['washout']
            if new_sum <= 0 and U[u_index] <= 0: # if the new sum is negative and the uninfected population is greater than 0, set it to 0
                new_U[u_index] = 0
            else:
                new_U[u_index] = new_sum

        # update I vector
        for infected_bacteria in bacteria_nodes:
            if (P[p_index] <= 0):
                continue
            i_index = bacteria_nodes.index(infected_bacteria)
            for k_index in range(0, M):
                if k_index == 0: # if we are at the first stage of infection
                    p_sum = 0
                    for phage in phage_nodes: # loop over the phage names
                        p_index = phage_nodes.index(phage) # get index of the phage
                        if graph.has_edge(phage, infected_bacteria): # check if the edge between phage_p and bacteria_b exists.
                            p_sum += r_matrix[p_index, i_index] * P[p_index]
                        M_tau = 0 if tau_vector[i_index] == 0 else M / tau_vector[i_index]
                    new_sum = U[i_index] * p_sum - M_tau * I[i_index, 0] - environment['washout'] * U[i_index]
                    if new_sum <= 0 and I[i_index, 0] <= 0: # if the new sum is negative and the infected population is greater than 0, set it to 0
                        new_I[i_index, 0] = 0
                    else:
                        new_I[i_index, 0] = new_sum
                else: # if we are at the other stages of infection
                    if(tau_vector[i_index] == 0): # prevent divide by 0 error
                        M_tau = 0
                    else:
                        m_tau = M / tau_vector[i_index] # get the value of M_tau
                    right = I[i_index, k_index - 1] - I[i_index, k_index]
                    new_sum = m_tau * right - environment['washout'] * I[i_index, k_index] 
                    if new_sum <= 0 and I[i_index, k_index] <= 0: # if the new sum is negative and the infected population is less than 0, set it to 0
                        new_I[i_index, k_index] = 0
                    else:
                        new_I[i_index, k_index] = new_sum
        
        # update P vector
        for phage in phage_nodes: # loop over the phage names
            p_index = phage_nodes.index(phage) # get index of the phage
            left_sum = 0 # initialize sums
            right_sum = 0
            for infected_bacteria in bacteria_nodes: # loop over the bacteria names
                i_index = bacteria_nodes.index(infected_bacteria) # get index of the bacteria
                if graph.has_edge(phage, infected_bacteria): # check if the edge between phage_p and bacteria_b exists.
                    if (tau_vector[i_index] == 0): # prevent divide by 0 error
                        M_tau = 0
                    else:
                        M_tau = M / tau_vector[i_index] # get the value of M_tau
                    left_sum += B_matrix[p_index, i_index] * M_tau * I[i_index, -1]
                    right_sum += r_matrix[p_index, i_index] * (U[i_index] + np.sum(I[i_index]))
            # update the phage value
            new_sum = left_sum - right_sum * P[p_index] - environment['washout'] * P[p_index]
            if new_sum <= 0 and P[p_index] <= 0: # if the new sum is negative and the phage population is greater than 0, set it to 0
                new_P[p_index] = 0
            else:
                new_P[p_index] = new_sum

        # flatten the new updated initial conditions, undoes the flattening done by unflatten_initial_matrix(). 
        flattened_y1 = self.flatten_lists_and_matrices(new_R, new_U, new_I, new_P)
        return flattened_y1


# graph = GraphMakerGUI(seed=0) # create a new object using the GUI tool. 
# system = System('simple_graph.gexf') # load the graph from the file.
# system = System('a_good_curve.gexf') # load the graph from the file.
# system = System('a_good_curve_2.gexf') # load the graph from the file.
# system = System('complex_graph.gexf') # load the graph from the file.
system = System('large_graph.gexf')

phage_nodes = system.get_nodes_of_type('P') # get the phage nodes
bacteria_nodes = system.get_nodes_of_type('B') # get the bacteria nodes
resource_nodes = system.get_nodes_of_type('R') # get the resource nodes
environment_nodes = system.get_nodes_of_type('E') # get the environment nodes

# get the 'Initial_Condition' attribute values from the nodes. Saves as vector. 
R0 = system.initialize_new_parameter_from_node("Initial_Concentration", resource_nodes)
U0 = system.initialize_new_parameter_from_node("Initial_Population", bacteria_nodes)
I0 = system.initialize_new_matrix(len(U0), system.M)
P0 = system.initialize_new_parameter_from_node("Initial_Population", phage_nodes)
# get the 'tau' and 'washin' values from the bacteria and resource nodes. Saves as vector
tau_vector = system.initialize_new_parameter_from_node('tau', bacteria_nodes)
washin = system.initialize_new_parameter_from_node('washin', resource_nodes)

# get the 'e', 'v', 'K', 'r', and 'B' values from the edges between the listed nodes. Saves as matrix.
e_matrix = system.initialize_new_parameter_from_edges('e', bacteria_nodes, resource_nodes)
v_matrix = system.initialize_new_parameter_from_edges('v', bacteria_nodes, resource_nodes)
K_matrix = system.initialize_new_parameter_from_edges('K', bacteria_nodes, resource_nodes)
r_matrix = system.initialize_new_parameter_from_edges('r', phage_nodes, bacteria_nodes)
B_matrix = system.initialize_new_parameter_from_edges('Burst_Size', phage_nodes, bacteria_nodes)

visualizer = Visualizer(system) # start the visualizer system. 

# add the initial conditions to the visualizer, with a name, the initial conditions, and the node names.
visualizer.add_graph_data("Resources", R0, resource_nodes)
# create an uninfected 'hidden' agent
visualizer.add_graph_data("Uninfected Bacteria", U0, bacteria_nodes)
# create infected 'hidden' agent. provide row names, as well as column names.
visualizer.add_graph_data("Infected Bacteria", I0, column_names=[f"Infected Stage {i}" for i in range(int(system.M))], row_names=[f"Infected B{i}" for i in range(len(bacteria_nodes))], add_rows=int(system.M))
visualizer.add_graph_data("Phages", P0, phage_nodes)

# add the vector parameters to the visualizer, with a name, the parameter values, and the node names.
visualizer.add_non_graph_data_vector("tau_vector", tau_vector, bacteria_nodes)
visualizer.add_non_graph_data_vector("washin", washin, resource_nodes)

# add matrix parameters to the visualizer, with a name, the parameter values, and the node names.
visualizer.add_non_graph_data_matrix("e_matrix", e_matrix, bacteria_nodes, resource_nodes)
visualizer.add_non_graph_data_matrix("v_matrix", v_matrix, bacteria_nodes, resource_nodes)
visualizer.add_non_graph_data_matrix("K_matrix", K_matrix, bacteria_nodes, resource_nodes)
visualizer.add_non_graph_data_matrix("r_matrix", r_matrix, phage_nodes, bacteria_nodes)
visualizer.add_non_graph_data_matrix("B_matrix", B_matrix, phage_nodes, bacteria_nodes)

# optionally add other parameters to the visualizer, that will be passed to the ODE system.
visualizer.add_other_parameters(phage_nodes, bacteria_nodes, resource_nodes, int(system.M))

visualizer.run() # run the visualizer/dashboard