import numpy as np
from Classes.Analysis import Analysis
from Classes.GraphMakerGUI import GraphMakerGUI
import pandas as pd
from dash import Dash, dash_table, html, Input, Output, callback, ALL, State, MATCH
from dash import dcc
import plotly.graph_objs as go

class Visualizer():
    def __init__(self, graph):
        self.app = Dash()
        self.graph = graph
        self.graph_data = {}
        self.non_graph_data_vector = {}
        self.non_graph_data_matrix = {}
        self.other_parameters_to_pass = []

    def add_graph_data(self, name, data, column_names, row_names = None, add_columns = False):
        self.graph_data[name] = {"data":data, "column_names":column_names, "row_names":row_names, "add_columns":add_columns}
    
    def add_non_graph_data_vector(self, name, data, column_names):
        self.non_graph_data_vector[name] = {"data":data, "column_names":column_names}
    
    def add_non_graph_data_matrix(self, name, data, row_names, column_names):
        self.non_graph_data_matrix[name] = {"data":data, "row_names":row_names, "column_names":column_names}

    def add_other_parameters(self, *args):
        self.other_parameters_to_pass += args

    def run(self):
        self.app.layout = html.Div([
            html.H1("Line Chart"),
            *[
            dcc.Graph(id={"type": "plotting-graph-data", "index": name}) for name in self.graph_data.keys()
            ],
            html.Button('Save and rerun model', id='submit-matrices'),
            dcc.Tabs([
                dcc.Tab(label='Graphing Data (Initial Conditions)', children=[
                    *[
                    html.Div([
                        html.H2(f"DataTable for {name}"),
                        html.H3(f"Row names {dic['row_names']}") if dic["row_names"] is not None else None,
                        dash_table.DataTable(
                        pd.DataFrame([dic["data"]], columns=dic["column_names"]).to_dict('records') if dic["row_names"] is None else pd.DataFrame(dic["data"], columns=dic["column_names"]).to_dict('records'),
                        id={"type":'edit-graphing-data', 'index': name},
                        columns=[{'name': f"{col}", 'id': col} for col in dic["column_names"]],
                        editable=True
                        ),
                    ]) for name, dic in self.graph_data.items()
                    ]
                ]),
                dcc.Tab(label='Non Graphing Data (Parameter Values): Vectors', children=[
                    *[
                    html.Div([
                        html.H2(f"DataTable for {name}"),
                        dash_table.DataTable(
                        pd.DataFrame([dic["data"]], columns=dic["column_names"]).to_dict('records'),
                        id={"type":'edit-non-graphing-data-vectors', 'index': name},
                        columns=[{'name': f"{col}", 'id': col} for col in dic["column_names"]],
                        editable=True
                        ),
                    ]) for name, dic in self.non_graph_data_vector.items()
                    ]
                ]),
                dcc.Tab(label='Non Graphing Data (Parameter Values): Matrices', children=[
                    *[
                    html.Div([
                        html.H2(f"DataTable for {name}"),
                        html.H3(f"Row Names: {dic['row_names']}"),
                        dash_table.DataTable(
                        pd.DataFrame(dic["data"], columns=dic["column_names"]).to_dict('records'),
                        id={"type":'edit-non-graphing-data-matrices', 'index': name},
                        columns=[{'name': f"{col}", 'id': col} for col in dic["column_names"]],
                        editable=True
                        ),
                    ]) for name, dic in self.non_graph_data_matrix.items()
                    ]
                ]),
                dcc.Tab(label='Environment Parameters', children=[
                    html.Div([
                    html.H2(f"DataTable for Environment Parameters"),
                    dash_table.DataTable(
                        pd.DataFrame([self.graph.environment_node_data]).to_dict('records'),
                        id={"type": 'environment variables', 'index': "environment variables"},
                        columns=[{"name": col, "id": col} for col in self.graph.environment_node_data.keys()],
                        editable=True
                    ),
                    ])
                ])
            ]),
            html.Div(style={'margin': '60px'}),
            html.H2(["Serial Transfer"]),
            dcc.Input(
                id="serial_transfer_value", type="number", placeholder="input with range",
                min=1, max=1_000_000, step=0.01, value=10
            ),
            html.Button("Run Serial Transfer", id="run_serial_transfer"),
            html.Div(style={'margin': '60px'}),
        ])

        @callback(
            [Output({'type': 'plotting-graph-data', 'index': name}, 'figure') for name in self.graph_data.keys()],
            Input('submit-matrices', 'n_clicks'),
            State({'type': 'edit-graphing-data', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-vectors', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-matrices', 'index': ALL}, 'data'), 
            State({'type': 'environment variables', 'index': "environment variables"}, 'data'),
            prevent_initial_call=True
        )
        def rerun_matrices(n_clicks, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            new_graphing_data = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data]
            flattened = self.graph.flatten_lists_and_matrices(*new_graphing_data)
            new_non_graphing_data_vectors = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy()[0] for data_values in graphing_data_vectors]
            new_non_graphing_data_matrices = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data_matrices]
            self.graph.add_environment_data(environment_data[0])
            print(*self.other_parameters_to_pass)
            new_updated_data = self.graph.solve_system(self.graph.odesystem, flattened, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices)
            solved_y = new_updated_data.y
            unflattened_data = self.graph.unflatten_initial_matrix(solved_y, [length["data"].size for length in self.graph_data.values()])
            
            new_unflattened_data = []
            for dic, unflattened in zip(self.graph_data.items(), unflattened_data):
                key, value = dic
                self.graph_data[key]["y_data"] = unflattened
                self.graph_data[key]["t_data"] = new_updated_data.t
                if value["add_columns"] not in [None, False]:
                    unflattened_temp = []
                    for i in range(0, len(unflattened), value["add_columns"]):
                        unflattened_temp.append(np.sum(unflattened[i:i+value["add_columns"]], axis=0))
                    new_unflattened_data.append(unflattened_temp)
                else:
                    new_unflattened_data.append(unflattened)
            unflattened_data = new_unflattened_data.copy()
            list_of_figs = []
            for i, dictionary in enumerate(self.graph_data.items()):
                name, dic = dictionary
                fig = go.Figure(dict(text=name))
                for j in range(len(unflattened_data[i])):
                    fig.add_trace(go.Scatter(x=new_updated_data.t, y=unflattened_data[i][j], mode="lines", name=f"{dic['column_names'][j]}"))
                    fig.update_layout(title=f"{name} vs Time", xaxis_title="Time", yaxis_title=name)
                list_of_figs.append(fig)
                i += 1
            return list_of_figs
        self.app.run_server(debug=True)