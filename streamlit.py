from os import system
import numpy as np
from Classes.Analysis import Analysis
from Classes.GraphMakerGUI import GraphMakerGUI
from Classes.Visualizer import Visualizer

class System(Analysis):
    def __init__(self, graph_location):
        super().__init__(graph_location)

    def odesystem(self, t, Y, *params):
        # start simple, bacteria-resource, see how the bacteria and resources grow/shrink, bacteria should hit carrying capacity, resource should reach 0, not negative, etc
        graph_object, phage_nodes, bacteria_nodes, resource_nodes, M, tau_vector, washin_vector, e_matrix, v_matrix, K_matrix, r_matrix, B_matrix, environment = params
        graph = graph_object.graph
        def g(N, v, K):
            return (N * v) / (N + K)

        Y = self.check_cutoff(Y)
        
        N, U, I, P = self.unflatten_initial_matrix(Y, [len(resource_nodes), len(bacteria_nodes), (len(bacteria_nodes), M), len(phage_nodes)])
        new_N = np.zeros_like(N)
        new_U = np.zeros_like(U)
        new_I = np.zeros_like(I)
        new_P = np.zeros_like(P)
        #update N vector
        for resource in resource_nodes:
            n_index = resource_nodes.index(resource)
            washin = washin_vector[n_index]
            sum = 0
            for bacteria in bacteria_nodes:
                b_index = bacteria_nodes.index(bacteria)
                if graph.has_edge(bacteria, resource):
                    e = e_matrix[b_index, n_index] 
                    v = v_matrix[b_index, n_index]
                    K = K_matrix[b_index, n_index]
                    U_b = U[b_index]
                    I_b = np.sum(I[b_index])
                    sum += e * g(N[n_index], v, K) * (U_b + I_b)
            new_N[n_index] = -sum - N[n_index] * environment['washout'] + washin
        
        # update U vector, i, and j are flipped relative to what is seen in update N vector for v, K, and r matrices because of how the row and columns are defined in the graph
        # dont sum U in left and right, because we are looking at an individual bacteria
        for uninfected in bacteria_nodes:
            u_index = bacteria_nodes.index(uninfected)
            p_sum = 0
            g_sum = 0
            for resource in resource_nodes:
                n_index = resource_nodes.index(resource)
                if graph.has_edge(uninfected, resource):
                    g_sum += g(N[n_index], v_matrix[u_index, n_index], K_matrix[u_index, n_index])
            for phage in phage_nodes:
                p_index = phage_nodes.index(phage)
                if graph.has_edge(phage, uninfected):
                    p_sum += r_matrix[p_index, u_index] * P[p_index]
            new_U[u_index] = U[u_index] * g_sum - U[u_index] * p_sum - U[u_index] * environment['washout']

        for infected in bacteria_nodes:
            i_index = bacteria_nodes.index(infected)
            for infected_stage in range(0, M):
                if infected_stage == 0:
                    p_sum = 0
                    for phage in phage_nodes:
                        p_index = phage_nodes.index(phage)
                        if graph.has_edge(phage, infected):
                            p_sum += r_matrix[p_index, i_index] * P[p_index]
                            if (tau_vector[i_index] == 0):
                                M_tau = 0
                            else:
                                M_tau = M / tau_vector[i_index]
                        else:
                            M_tau = 0
                    new_I[i_index, 0] = U[i_index] * p_sum - M_tau * I[i_index, 0] - environment['washout'] * U[i_index]
                else:
                    if(tau_vector[i_index] == 0):
                        M_tau = 0
                    else:
                        m_tau = M / tau_vector[i_index]
                    right = I[i_index, infected_stage - 1] - I[i_index, infected_stage]
                    new_I[i_index, infected_stage] = m_tau * right - environment['washout'] * new_I[i_index, infected_stage] 
        
        for phage in phage_nodes:
            p_index = phage_nodes.index(phage)
            left_sum = 0
            right_sum = 0
            for infected in bacteria_nodes:
                i_index = bacteria_nodes.index(infected)
                if graph.has_edge(phage, infected):
                    if (tau_vector[i_index] == 0):
                        M_tau = 0
                    else:
                        M_tau = M / tau_vector[i_index]
                    left_sum += B_matrix[p_index, i_index] * M_tau * I[i_index, -1]
                    right_sum += r_matrix[p_index, i_index] * (U[i_index] + np.sum(I[i_index]))
            new_P[p_index] = left_sum - right_sum * P[p_index] - environment['washout'] * P[p_index]

        flattened_y1 = self.flatten_lists_and_matrices(new_N, new_U, new_I, new_P)
        flattened_y1 = self.prevent_negative_numbers(Y, flattened_y1)
        print(flattened_y1)
        return flattened_y1


# graph = GraphMakerGUI(seed=0)
system = System('simple_graph.gexf')
# system = System('complex_graph.gexf')

phage_nodes = system.get_nodes_of_type('P')
bacteria_nodes = system.get_nodes_of_type('B')
resource_nodes = system.get_nodes_of_type('R')
environment_nodes = system.get_nodes_of_type('E')

R0 = system.initialize_new_parameter_from_node("Initial_Concentration", resource_nodes)
U0 = system.initialize_new_parameter_from_node("Initial_Population", bacteria_nodes)
I0 = system.initialize_new_matrix(len(U0), system.M)
P0 = system.initialize_new_parameter_from_node("Initial_Population", phage_nodes)
tau_vector = system.initialize_new_parameter_from_node('tau', bacteria_nodes)
washin = system.initialize_new_parameter_from_node('washin', resource_nodes)

e_matrix = system.initialize_new_parameter_from_edges('e', bacteria_nodes, resource_nodes)
v_matrix = system.initialize_new_parameter_from_edges('v', bacteria_nodes, resource_nodes)
K_matrix = system.initialize_new_parameter_from_edges('K', bacteria_nodes, resource_nodes)
r_matrix = system.initialize_new_parameter_from_edges('r', phage_nodes, bacteria_nodes)
B_matrix = system.initialize_new_parameter_from_edges('Burst_Size', phage_nodes, bacteria_nodes)

visualizer = Visualizer(system)
visualizer.add_graph_data("Resources", R0, resource_nodes)
visualizer.add_graph_data("Uninfected Bacteria", U0, bacteria_nodes)
visualizer.add_graph_data("Infected Bacteria", I0, row_names=bacteria_nodes, column_names=[f"Infected B{i}" for i in range(int(4))], add_rows=4)
visualizer.add_graph_data("Phages", P0, phage_nodes)

visualizer.add_non_graph_data_vector("tau_vector", tau_vector, bacteria_nodes)
visualizer.add_non_graph_data_vector("washin", washin, resource_nodes)

visualizer.add_non_graph_data_matrix("e_matrix", e_matrix, bacteria_nodes, resource_nodes)
visualizer.add_non_graph_data_matrix("v_matrix", v_matrix, bacteria_nodes, resource_nodes)
visualizer.add_non_graph_data_matrix("K_matrix", K_matrix, bacteria_nodes, resource_nodes)
visualizer.add_non_graph_data_matrix("r_matrix", r_matrix, phage_nodes, bacteria_nodes)
visualizer.add_non_graph_data_matrix("B_matrix", B_matrix, phage_nodes, bacteria_nodes)

visualizer.add_other_parameters(phage_nodes, bacteria_nodes, resource_nodes, int(system.M))

visualizer.run()