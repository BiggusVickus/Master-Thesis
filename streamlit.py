import numpy as np
from Classes.Analysis import Analysis
from Classes.GraphMakerGUI import GraphMakerGUI
import pandas as pd
from dash import Dash, dash_table, html, Input, Output, callback, ALL, State, MATCH
from dash import dcc
import plotly.graph_objs as go
# from Classes.StreamlitVisualization import StreamlitVisualization

class System(Analysis):
    def __init__(self, graph_location):
        super().__init__(graph_location)

    def new_system2(self, t, Y, *params):
        # TODO: rewrite this into for loops
        #TODO: look at biology side, try to replicate graphs
        #TODO: explore the model(s)
        # start simple, bacteria-resource, see how the bacteria and reosurces grow/shrink, bacteria should hit carrying capacity, nutrient should reach 0, not negative, etc
        graph_object, phage_nodes, bacteria_nodes, nutrient_nodes, e_vector, tau_vector, v_matrix, K_matrix, r_matrix, B_matrix, M = params
        graph = graph_object.graph
        def g(N, v, K):
            return (N * v) / (N + K)
        Y = self.check_cutoff(Y, 0.000001)
        
        N, U, I, P = self.unflatten_initial_matrix(Y, [len(nutrient_nodes), len(bacteria_nodes), (len(bacteria_nodes), M), len(phage_nodes)])
        new_N = np.zeros_like(N)
        new_U = np.zeros_like(U)
        new_I = np.zeros_like(I)
        new_P = np.zeros_like(P)
        #update N vector
        for nutrient in nutrient_nodes:
            n_index = resource_nodes.index(nutrient)
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
            new_N[n_index] = (e_value * sum_g) * (sum_u + sum_i)
        
        # update U vector, i, and j are flipped relative to what is seen in update N vector for v, K, and r matrices because of how the row and columns are defined in the graph
        # dont sum U in left and right, because we are looking at an individual bacteria
        for uninfected in bacteria_nodes:
            u_index = bacteria_nodes.index(uninfected)
            g_sum = 0
            right = 0
            for nutrient in nutrient_nodes:
                n_index = resource_nodes.index(nutrient)
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

graph = System('simple_test.gexf')
# graph = System('example.gexf')
phage_nodes = graph.get_nodes_of_type('P')
bacteria_nodes = graph.get_nodes_of_type('B')
resource_nodes = graph.get_nodes_of_type('R')
environemnt_nodes = graph.get_nodes_of_type('E')

R0 = graph.initialize_new_parameter_from_node(resource_nodes, "Initial_Concentration")
U0 = graph.initialize_new_parameter_from_node(bacteria_nodes, "Initial_Population")
I0 = graph.initialize_new_matrix(len(U0), graph.M)
P0 = graph.initialize_new_parameter_from_node(phage_nodes, "Initial_Population")

e_vector = graph.initialize_new_parameter_from_node(resource_nodes, 'e')
v_matrix = graph.initialize_new_parameter_from_edges(bacteria_nodes, resource_nodes, 'v')
K_matrix = graph.initialize_new_parameter_from_edges(bacteria_nodes, resource_nodes, 'K')
r_matrix = graph.initialize_new_parameter_from_edges(phage_nodes, bacteria_nodes, 'r')
tau_vector = graph.initialize_new_parameter_from_node(bacteria_nodes, 'tau')
B_matrix = graph.initialize_new_parameter_from_edges(phage_nodes, bacteria_nodes, 'Burst_Size')

graph_data = {"R0": pd.DataFrame(R0, columns=["R0 0"]), "U0": pd.DataFrame(U0, columns=["U0 0"]), "I0": pd.DataFrame(I0, columns=["I0 0", "I0 1", "I0 2", "I0 3"]), "P0": pd.DataFrame(P0, columns=["P0 0"])}
non_graph_data_vector = {"e_vector": pd.DataFrame(e_vector, columns=["e_vector 0"]), "tau_vector": pd.DataFrame(tau_vector, columns=["tau_vector 0"])}
non_graph_data_matrix = {"v_matrix": pd.DataFrame(v_matrix, columns=["v_matrix 0"]), "K_matrix": pd.DataFrame(K_matrix, columns=["K_matrix 0"]), "r_matrix": pd.DataFrame(r_matrix, columns=["r_matrix 0"]), "B_matrix": pd.DataFrame(B_matrix, columns=["B_matrix 0"])}

# graph_data = {"R0": pd.DataFrame(R0, columns=["R0 0"]), 
#     "U0": pd.DataFrame(U0, columns=["U0 0"]), "I0": 
#               pd.DataFrame(I0, columns=["I0 0", "I0 1", "I0 2", "I0 3"]), 
#               "P0": pd.DataFrame(P0, columns=["P0 0"])}
# non_graph_data_vector = {"e_vector": pd.DataFrame(e_vector, columns=["e_vector 0"]), "tau_vector": pd.DataFrame(tau_vector, columns=["tau_vector 0"])}
# non_graph_data_matrix = {"v_matrix": pd.DataFrame(v_matrix, columns=["v_matrix 0", "v_matrix 1", "v_matrix 2"]), 
#                          "K_matrix": pd.DataFrame(K_matrix, columns=["K_matrix 0", "K_matrix 1", "K_matrix 2"]), 
#                          "r_matrix": pd.DataFrame(r_matrix, columns=["r_matrix 0", "r_matrix 1", "r_matrix 2"]), 
#                          "B_matrix": pd.DataFrame(B_matrix, columns=["B_matrix 0", "B_matrix 1", "B_matrix 2"])}

app = Dash()
app.layout = [
    html.H1("Line Chart of N, U, I, and P"),
    html.Button('Save and rerun model', id='submit-matrices'),
    *[
        dcc.Graph(id={"type": "plotting-graph-data", "index": "R"}),
        dcc.Graph(id={"type": "plotting-graph-data", "index": "U"}),
        dcc.Graph(id={"type": "plotting-graph-data", "index": "I"}),
        dcc.Graph(id={"type": "plotting-graph-data", "index": "P"}),
    ],
    html.H1("Graphing Data"),
    *[
        html.Div([
            html.H2(f"DataTable for {name}"),
            dash_table.DataTable(
                table.to_dict('records'),
                id={"type":'edit-graphing-data', 'index': name},
                columns=[{'name': f"{col}", 'id': col} for col in table.columns],
                editable=True
            ),
        ]) for name, table in graph_data.items()
    ], 
    html.H1("Non Graphing Data: Vectors"),
    *[
        html.Div([
            html.H2(f"DataTable for {name}"),
            dash_table.DataTable(
                table.to_dict('records'),
                id={"type":'edit-non-graphing-data-vectors', 'index': name},
                columns=[{'name': f"{col}", 'id': col} for col in table.columns],
                editable=True
            ),
        ]) for name, table in non_graph_data_vector.items()
    ], 
    html.H1("Non Graphing Data: Matrices"),
    *[
        html.Div([
            html.H2(f"DataTable for {name}"),
            dash_table.DataTable(
                table.to_dict('records'),
                id={"type":'edit-non-graphing-data-matrices', 'index': name},
                columns=[{'name': f"{col}", 'id': col} for col in table.columns],
                editable=True
            ),
        ]) for name, table in non_graph_data_matrix.items()
    ], 
    dcc.Graph(id={"type": "line-chart", "index": 1}),  # Dynamic ID
]


@callback(
    [Output({"type": "plotting-graph-data", "index": "R"}, "figure"),
    Output({"type": "plotting-graph-data", "index": "U"}, "figure"),
    Output({"type": "plotting-graph-data", "index": "I"}, "figure"),
    Output({"type": "plotting-graph-data", "index": "P"}, "figure")],
    Input('submit-matrices', 'n_clicks'),
    State({'type': 'edit-graphing-data', 'index': ALL}, 'data'),
    State({'type': 'edit-non-graphing-data-vectors', 'index': ALL}, 'data'),
    State({'type': 'edit-non-graphing-data-matrices', 'index': ALL}, 'data'), prevent_initial_call=True
)
def rerun_matrices(n_clicks, graphing_data, graphing_data_vectors, graphing_data_matrices):
    new_graphing_data = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data]
    flattened = graph.flatten_lists_and_matrices(*new_graphing_data)
    new_non_graphing_data_vectors = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy().T[0] for data_values in graphing_data_vectors]
    new_non_graphing_data_matrices = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data_matrices]
    new_updated_data = graph.solve_system(graph.new_system2, flattened, graph, phage_nodes, bacteria_nodes, resource_nodes, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices, int(graph.M))
    solved_y = new_updated_data.y
    new_N, new_U, new_I, new_P = graph.unflatten_initial_matrix(solved_y, [len(R0), len(U0), 4, len(P0)])
    new_I = [new_I[0] + new_I[1] + new_I[2] + new_I[3]]
    fig1 = go.Figure()
    for i in range(len(new_N)):
        fig1.add_trace(go.Scatter(x=new_updated_data.t, y=new_N[i], mode="lines", name="Updated Line"))
    fig2 = go.Figure()
    for i in range(len(new_U)):
        fig2.add_trace(go.Scatter(x=new_updated_data.t, y=new_U[i], mode="lines", name="Updated Line"))
    fig3 = go.Figure()
    for i in range(len(new_I)):
        fig3.add_trace(go.Scatter(x=new_updated_data.t, y=new_I[i], mode="lines", name="Updated Line"))
    fig4 = go.Figure()
    for i in range(len(new_P)):
        fig4.add_trace(go.Scatter(x=new_updated_data.t, y=new_P[i], mode="lines", name="Updated Line"))
    return [fig1, fig2, fig3, fig4]

app.run(debug=True, use_reloader=True)