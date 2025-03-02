import numpy as np
from Classes.Analysis import Analysis
from Classes.GraphMakerGUI import GraphMakerGUI
import pandas as pd
from dash import Dash, dash_table, html, Input, Output, callback, ALL, State, MATCH
from dash import dcc
import plotly.graph_objs as go
from collections import OrderedDict
from copy import deepcopy

class Visualizer():
    def __init__(self, graph):
        self.app = Dash()
        self.graph:Analysis = graph
        self.graph_data = OrderedDict()
        self.non_graph_data_vector = OrderedDict()
        self.non_graph_data_matrix = OrderedDict()
        self.other_parameters_to_pass = []
        self.copy_of_simulation_output = None

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
            # [Output({'type': 'plotting-graph-data', 'index': name}, 'figure') for name in self.graph_data.keys()],
            Input('submit-matrices', 'n_clicks'),
            State({'type': 'edit-graphing-data', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-vectors', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-matrices', 'index': ALL}, 'data'), 
            State({'type': 'environment variables', 'index': "environment variables"}, 'data'),
            prevent_initial_call=False
        )
        def rerun_matrices(n_clicks, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            new_graphing_data = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data]
            flattened = self.graph.flatten_lists_and_matrices(*new_graphing_data)
            new_non_graphing_data_vectors = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy()[0] for data_values in graphing_data_vectors]
            new_non_graphing_data_matrices = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data_matrices]
            self.graph.add_environment_data(environment_data[0])
            new_updated_data = self.graph.solve_system(self.graph.odesystem, flattened, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices)
            solved_y = new_updated_data.y
            self.copy_of_simulation_output = new_updated_data
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
            # return list_of_figs
        
        @callback(
            [Output({'type': 'plotting-graph-data', 'index': name}, 'figure') for name in self.graph_data.keys()],
            Input('run_serial_transfer', 'n_clicks'),
            State('serial_transfer_value', 'value'),
            State({'type': 'edit-graphing-data', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-vectors', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-matrices', 'index': ALL}, 'data'), 
            State({'type': 'environment variables', 'index': "environment variables"}, 'data'),
            prevent_initial_call=True
        )
        def serial_transfer(n_clicks, serial_transfer, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            new_graphing_data = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data]
            flattened = self.graph.flatten_lists_and_matrices(*new_graphing_data)
            new_non_graphing_data_vectors = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy()[0] for data_values in graphing_data_vectors]
            new_non_graphing_data_matrices = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data_matrices]
            self.graph.add_environment_data(environment_data[0])
            original_time = self.copy_of_simulation_output.t
            original_final_time = self.copy_of_simulation_output.t[-1]
            original_simulation_output = self.copy_of_simulation_output.y
            original_final_simulation_output = self.copy_of_simulation_output.y[:,-1]

            row_of_names = []
            row_of_values = []
            for key, value in self.graph_data.items():
                row_of_names += [key] * value["data"].size

            for final, name, flat in zip(original_final_simulation_output, row_of_names, flattened):
                if (name.lower() in ["resources", "resource", "r", "res", "r0", "nutrient", "nutrients", "n", "nut", "n0"]):
                    row_of_values.append(flat + final/serial_transfer) 
                else:
                    row_of_values.append(final/serial_transfer)
            print(original_final_simulation_output)
            print('----')
            print(row_of_values)

            new_updated_data = self.graph.solve_system(self.graph.odesystem, row_of_values, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices, t_start = original_final_time, t_end=float(original_final_time)+float(self.graph.Simulation_Length))
            
            solved_y = new_updated_data.y
            print(original_time)
            print('----')
            print(new_updated_data.t)
            print('----')
            new_overall_t = np.concatenate((original_time, new_updated_data.t))
            print(new_overall_t)
            unflattened_data = self.graph.unflatten_initial_matrix(solved_y, [length["data"].size for length in self.graph_data.values()])
            new_unflattened_data = []
            for dic, unflattened in zip(self.graph_data.items(), unflattened_data):
                key, value = dic
                # Append the new unflattened data to the existing data
                self.graph_data[key]["y_data"] = np.concatenate((self.graph_data[key]["y_data"], unflattened), axis=1)
                self.graph_data[key]["t_data"] = new_overall_t
                if value["add_columns"] not in [None, False]:
                    unflattened_temp = []
                    for i in range(0, len(self.graph_data[key]["y_data"]), value["add_columns"]):
                        unflattened_temp.append(np.sum(self.graph_data[key]["y_data"][i:i+value["add_columns"]], axis=0))
                    new_unflattened_data.append(unflattened_temp)
                else:
                    new_unflattened_data.append(self.graph_data[key]["y_data"])
            unflattened_data = new_unflattened_data.copy()
            list_of_figs = []
            for i, dictionary in enumerate(self.graph_data.items()):
                name, dic = dictionary
                fig = go.Figure(dict(text=name))
                for j in range(len(unflattened_data[i])):
                    fig.add_trace(go.Scatter(x=new_overall_t, y=unflattened_data[i][j], mode="lines", name=f"{dic['column_names'][j]}"))
                list_of_figs.append(fig)
            return list_of_figs
        
        self.app.run_server(debug=True)