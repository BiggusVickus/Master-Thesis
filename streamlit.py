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

# graph = System('simple_test.gexf')
graph = System('example.gexf')
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

# graph_data = {"R0": pd.DataFrame(R0, columns=["R0 0"]), "U0": pd.DataFrame(U0, columns=["U0 0"]), "I0": pd.DataFrame(I0, columns=["I0 0", "I0 1", "I0 2", "I0 3"]), "P0": pd.DataFrame(P0, columns=["P0 0"])}
# non_graph_data_vector = {"e_vector": pd.DataFrame(e_vector, columns=["e_vector 0"]), "tau_vector": pd.DataFrame(tau_vector, columns=["tau_vector 0"])}
# non_graph_data_matrix = {"v_matrix": pd.DataFrame(v_matrix, columns=["v_matrix 0"]), "K_matrix": pd.DataFrame(K_matrix, columns=["K_matrix 0"]), "r_matrix": pd.DataFrame(r_matrix, columns=["r_matrix 0"]), "B_matrix": pd.DataFrame(B_matrix, columns=["B_matrix 0"])}

graph_data = {"R0": pd.DataFrame(R0, columns=["R0 0"]), 
    "U0": pd.DataFrame(U0, columns=["U0 0"]), "I0": 
              pd.DataFrame(I0, columns=["I0 0", "I0 1", "I0 2", "I0 3"]), 
              "P0": pd.DataFrame(P0, columns=["P0 0"])}
non_graph_data_vector = {"e_vector": pd.DataFrame(e_vector, columns=["e_vector 0"]), "tau_vector": pd.DataFrame(tau_vector, columns=["tau_vector 0"])}
non_graph_data_matrix = {"v_matrix": pd.DataFrame(v_matrix, columns=["v_matrix 0", "v_matrix 1", "v_matrix 2"]), 
                         "K_matrix": pd.DataFrame(K_matrix, columns=["K_matrix 0", "K_matrix 1", "K_matrix 2"]), 
                         "r_matrix": pd.DataFrame(r_matrix, columns=["r_matrix 0", "r_matrix 1", "r_matrix 2"]), 
                         "B_matrix": pd.DataFrame(B_matrix, columns=["B_matrix 0", "B_matrix 1", "B_matrix 2"])}

app = Dash()
app.layout = html.Div([
    html.H1("Line Chart of new_U"),
    dcc.Graph(
        id='line-chart-new-N',
        figure={
            'data': [
                {
                    'x': list(range(len(new_N[i]))),
                    'y': new_N[i],
                    'type': 'line',
                    'name': 'new_N'
                } for i in range(len(new_N))
            ],
            'layout': {
                'title': 'Line Chart of new_N'
            }
        }
    ),
    dcc.Graph(
        id='line-chart-new-U',
        figure={
            'data': [
                {
                    'x': list(range(len(new_U[i]))),
                    'y': new_U[i],
                    'type': 'line',
                    'name': 'new_U'
                } for i in range(len(new_U))
            ],
            'layout': {
                'title': 'Line Chart of new_U'
            }
        }
    ),
    dcc.Graph(
        id='line-chart-new-I',
        figure={
            'data': [
                {
                    'x': list(range(len(new_I[i]))),
                    'y': new_I[i],
                    'type': 'line',
                    'name': 'new_I'
                } for i in range(len(new_I))
            ],
            'layout': {
                'title': 'Line Chart of new_I'
            }
        }
    ),
    dcc.Graph(
        id='line-chart-new-P',
        figure={
            'data': [
                {
                    'x': list(range(len(new_P[i]))),
                    'y': new_P[i],
                    'type': 'line',
                    'name': 'new_P'
                } for i in range(len(new_P))
            ],
            'layout': {
                'title': 'Line Chart of new_P'
            }
        }
    ),
    *[
        html.Div([
            html.H2(f"DataTable for {name}"),
            dash_table.DataTable(
                table.to_dict('records'),
                id={"type":'editing-matrix-data', 'index': name},
                editable=True, 
                style_data={"maxWidth": "100px", "overflow": "hidden", "textOverflow": "ellipsis"},
            ),
            html.Div(id=f'editing-prune-data-output-{name}')
        ]) for name, table in zip(list_name, list_tables)
    ], 
    html.Button('Submit', id='submit-matrices'),
])


@callback([
    Input('submit-matrices', 'n_clicks'),
    Input({'type': 'editing-matrix-data', 'index': ALL}, 'data')
]
)
def rerun_matrices(n_clicks, data):
    list_of_new_matrices = [np.array([[value for value in row.values()] for row in data_values]) for data_values in data]
    print(list_of_new_matrices)
    new_R0 = list_of_new_matrices[0]
    new_U0 = list_of_new_matrices[1]
    new_I0 = list_of_new_matrices[2]
    new_P0 = list_of_new_matrices[3]
    flattened = System.flatten_lists_and_matrices(new_N, new_U, new_I, new_P)
    rest = list_of_new_matrices[4:]
    print(flattened)
    data = graph.solve_system(graph.new_system2, flattened, *rest, int(graph.M), graph.max_length)
    print(data.y)
app.run(debug=True, use_reloader=True)

from dash import Dash, dash_table
import pandas as pd

df = pd.read_csv('test.csv')

app = Dash()

app.layout = dash_table.DataTable(df.to_dict('records'))
print(df.to_dict('records'))
if __name__ == '__main__':
    app.run(debug=True)