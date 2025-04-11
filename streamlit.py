import numpy as np
from Classes.Analysis import Analysis
from Classes.GraphMakerGUI import GraphMakerGUI
from Classes.Visualizer import Visualizer

class System(Analysis):
    def __init__(self, graph_location):
        super().__init__(graph_location)

    def odesystem(self, t, Y, *params):
        # start simple, bacteria-resource, see how the bacteria and reosurces grow/shrink, bacteria should hit carrying capacity, nutrient should reach 0, not negative, etc
        graph_object, phage_nodes, bacteria_nodes, nutrient_nodes, M, e_vector, tau_vector, v_matrix, K_matrix, r_matrix, B_matrix = params
        graph = graph_object.graph
        def g(N, v, K):
            return (N * v) / (N + K)

        Y = self.check_cutoff(Y)
        
        N, U, I, P = self.unflatten_initial_matrix(Y, [len(nutrient_nodes), len(bacteria_nodes), (len(bacteria_nodes), M), len(phage_nodes)])
        new_N = np.zeros_like(N)
        new_U = np.zeros_like(U)
        new_I = np.zeros_like(I)
        new_P = np.zeros_like(P)
        #update N vector
        for nutrient in nutrient_nodes:
            n_index = nutrient_nodes.index(nutrient)
            e_value = e_vector[n_index] 
            sum_g = 0
            sum_u = 0
            sum_i = 0
            for bacteria in bacteria_nodes:
                b_index = bacteria_nodes.index(bacteria)
                if graph.has_edge(bacteria, nutrient):
                    v = v_matrix[b_index, n_index]
                    K = K_matrix[b_index, n_index]
                    sum_g += g(N[n_index], v, K)
                    sum_u += U[b_index]
                    sum_i += np.sum(I[b_index])
            new_N[n_index] = -(e_value * sum_g) * (sum_u + sum_i)
        
        # update U vector, i, and j are flipped relative to what is seen in update N vector for v, K, and r matrices because of how the row and columns are defined in the graph
        # dont sum U in left and right, because we are looking at an individual bacteria
        for uninfected in bacteria_nodes:
            u_index = bacteria_nodes.index(uninfected)
            g_sum = 0
            right = 0
            for nutrient in nutrient_nodes:
                n_index = nutrient_nodes.index(nutrient)
                if graph.has_edge(uninfected, nutrient):
                    g_sum += g(N[n_index], v_matrix[u_index, n_index], K_matrix[u_index, n_index])
            for phage in phage_nodes:
                p_index = phage_nodes.index(phage)
                if graph.has_edge(phage, uninfected):
                    right += r_matrix[p_index, u_index] * P[p_index]
            new_U[u_index] = g_sum * U[u_index] - right * U[u_index]

        for infected in bacteria_nodes:
            i_index = bacteria_nodes.index(infected)
            for infected_stage in range(0, M):
                if infected_stage == 0:
                    left_sum = 0
                    right_sum = 0
                    for phage in phage_nodes:
                        p_index = phage_nodes.index(phage)
                        if graph.has_edge(phage, infected):
                            left_sum += r_matrix[p_index, i_index] * P[p_index]
                            right_sum += M / tau_vector[i_index] * I[i_index, 0]
                    new_I[i_index, 0] = left_sum * U[i_index] - right_sum
                else:
                    m_tau = M / tau_vector[i_index]
                    right = I[i_index, infected_stage - 1] - I[i_index, infected_stage]
                    new_I[i_index, infected_stage] = m_tau * right
        
        for phage in phage_nodes:
            p_index = phage_nodes.index(phage)
            left_sum = 0
            right_sum = 0
            for infected in bacteria_nodes:
                i_index = bacteria_nodes.index(infected)
                if graph.has_edge(phage, infected):
                    left_sum += B_matrix[p_index, i_index] * M / tau_vector[i_index] * I[i_index, -1]
                    right_sum += r_matrix[p_index, i_index] * (U[i_index] + np.sum(I[i_index])) * P[p_index]
            new_P[p_index] = left_sum - right_sum

        flattened_y1 = self.flatten_lists_and_matrices(new_N, new_U, new_I, new_P)
        return flattened_y1


# graph = GraphMakerGUI()
# graph.export_graph('simple_test.gexf')
# graph = System('simple_test.gexf')

# graph = System('example.gexf')
system = System('simple_test_2.gexf')
# graph = System('example_3.gexf')
# system.add_item_to_class_attribute('M', 4) # add the M value to the system

phage_nodes = system.get_nodes_of_type('P')
bacteria_nodes = system.get_nodes_of_type('B')
resource_nodes = system.get_nodes_of_type('R')
environemnt_nodes = system.get_nodes_of_type('E')

R0 = system.initialize_new_parameter_from_node(resource_nodes, "Initial_Concentration")
U0 = system.initialize_new_parameter_from_node(bacteria_nodes, "Initial_Population")
I0 = system.initialize_new_matrix(len(U0), system.M)
P0 = system.initialize_new_parameter_from_node(phage_nodes, "Initial_Population")

e_vector = system.initialize_new_parameter_from_node(resource_nodes, 'e')
tau_vector = system.initialize_new_parameter_from_node(bacteria_nodes, 'tau')
v_matrix = system.initialize_new_parameter_from_edges(bacteria_nodes, resource_nodes, 'v')
K_matrix = system.initialize_new_parameter_from_edges(bacteria_nodes, resource_nodes, 'K')
r_matrix = system.initialize_new_parameter_from_edges(phage_nodes, bacteria_nodes, 'r')
B_matrix = system.initialize_new_parameter_from_edges(phage_nodes, bacteria_nodes, 'Burst_Size')

visualizer = Visualizer(system)
visualizer.add_graph_data("Resources", R0, resource_nodes)
visualizer.add_graph_data("Uninfected Bacteria", U0, bacteria_nodes)
visualizer.add_graph_data("Infected Bacteria", I0, row_names=bacteria_nodes, column_names=[f"Infected B{i}" for i in range(int(system.M))], add_rows=4)
visualizer.add_graph_data("Phages", P0 , phage_nodes)

visualizer.add_non_graph_data_vector("e_vector", e_vector, resource_nodes)
visualizer.add_non_graph_data_vector("tau_vector", tau_vector, bacteria_nodes)
visualizer.add_non_graph_data_matrix("v_matrix", v_matrix, bacteria_nodes, resource_nodes)
visualizer.add_non_graph_data_matrix("K_matrix", K_matrix, bacteria_nodes, resource_nodes)
visualizer.add_non_graph_data_matrix("r_matrix", r_matrix, phage_nodes, bacteria_nodes)
visualizer.add_non_graph_data_matrix("B_matrix", B_matrix, phage_nodes, bacteria_nodes)

visualizer.add_other_parameters(phage_nodes, bacteria_nodes, resource_nodes, int(system.M))

visualizer.run()