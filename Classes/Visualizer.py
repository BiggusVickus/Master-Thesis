import numpy as np
from Classes.Analysis import Analysis
from Classes.GraphMakerGUI import GraphMakerGUI
import plotly.express as px
import pandas as pd
from dash import Dash, dash_table, html, Input, Output, callback, ALL, State, MATCH
from dash import dcc
import plotly.graph_objs as go
from collections import OrderedDict
from copy import deepcopy

class Visualizer():
    def __init__(self, graph):
        self.app = Dash()
        self.graph: Analysis = graph
        self.graph_data = OrderedDict()
        self.non_graph_data_vector = OrderedDict()
        self.non_graph_data_matrix = OrderedDict()
        self.other_parameters_to_pass = []
        self.copy_of_simulation_output = None

    def add_graph_data(self, name, data, column_names, row_names=None, add_rows=False):
        self.graph_data[name] = {"data": data, "column_names": column_names, "row_names": row_names, "add_rows": add_rows}

    def add_non_graph_data_vector(self, name, data, column_names):
        self.non_graph_data_vector[name] = {"data": data, "column_names": column_names}

    def add_non_graph_data_matrix(self, name, data, row_names, column_names):
        self.non_graph_data_matrix[name] = {"data": data, "row_names": row_names, "column_names": column_names}

    def add_other_parameters(self, *args):
        self.other_parameters_to_pass += args
    
    def create_figures(self, unflattened_data, new_overall_t):
        list_of_figs = []
        for i, dictionary in enumerate(self.graph_data.items()):
            name, dic = dictionary
            fig = go.Figure(dict(text=name))
            for j in range(len(unflattened_data[i])):
                fig.add_trace(go.Scatter(x=new_overall_t, y=unflattened_data[i][j], mode="lines", name=f"{dic['column_names'][j]}"))
                fig.update_layout(
                    title=f"Graph for {name}",
                    xaxis=dict(title="Time"),
                    yaxis=dict(title="Value")
                )
            list_of_figs.append(fig)
        return list_of_figs
        
    def create_numpy_lists(self, graphing_data, graphing_data_vectors, graphing_data_matrices):
        new_graphing_data = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data]
        flattened = self.graph.flatten_lists_and_matrices(*new_graphing_data)
        new_non_graphing_data_vectors = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy()[0] for data_values in graphing_data_vectors]
        new_non_graphing_data_matrices = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data_matrices]
        return new_graphing_data, flattened, new_non_graphing_data_vectors, new_non_graphing_data_matrices
    
    def serial_transfer_calculation(self, original_final_simulation_output, serial_transfer_value, serial_tranfer_option, flattened):
        row_of_names = []
        row_of_values = []
        for key, value in self.graph_data.items():
            row_of_names += [key] * value["data"].size

        if (len(serial_tranfer_option) > 0):
            return flattened + original_final_simulation_output / serial_transfer_value
        for final, name, flat in zip(original_final_simulation_output, row_of_names, flattened):
            if (name.lower() in ["resources", "resource", "r", "res", "r0", "nutrient", "nutrients", "n", "nut", "n0"]):
                row_of_values.append(flat + final / serial_transfer_value)
            else:
                row_of_values.append(final / serial_transfer_value)
    
    def sum_up_columns(self, unflattened_data, value_add_column):
        if value_add_column not in [None, False]:
            unflattened_temp = []
            for i in range(0, len(unflattened_data), value_add_column):
                unflattened_temp.append(np.sum(unflattened_data[i:i + value_add_column], axis=0))
            return unflattened_temp
        else:
            return unflattened_data
        
    def save_data(self, unflattened_data, time, save_data=True):
        new_unflattened_data = []
        for dic, unflattened in zip(self.graph_data.items(), unflattened_data):
            key, value = dic
            if save_data:
                self.graph_data[key]["y_data"] = unflattened
                self.graph_data[key]["t_data"] = time
            new_unflattened_data.append(self.sum_up_columns(unflattened, value["add_rows"]))
        return new_unflattened_data
    
    def create_heatmap(self, data, x_axis_data, y_axis_data, x_labels, y_labels, title):
        df = pd.DataFrame(data, columns=y_axis_data, index=x_axis_data)
        fig = px.imshow(
            df, 
            labels={'x': y_labels, 'y': x_labels}, 
            text_auto=True, 
            aspect="equal" 
        )
        fig.update_layout(
            title=title,
            xaxis=dict(
                # scaleanchor="x",
                tickmode="array",
                tickvals=list(range(len(y_axis_data))), 
                ticktext=y_axis_data, 
                categoryarray=y_axis_data,      
            ),
            yaxis=dict(
                # scaleanchor="y",
                tickmode="array",
                tickvals=list(range(len(x_axis_data))), 
                ticktext=x_axis_data,
                categoryarray=x_axis_data,
            ), 
        )
        fig.update_traces(
            hovertemplate=f"{y_labels}: %{{x}}<br>{x_labels}: %{{y}}<br>End Value: %{{z}}<extra></extra>"
        )
        fig.update_xaxes(type='category')
        fig.update_yaxes(type='category')
        return fig
        
    def run(self):
        param_initial = [name for name in self.graph_data.keys()]
        params = [name for name in OrderedDict(list(self.non_graph_data_vector.items()) + list(self.non_graph_data_matrix.items()))]
        both_params = param_initial + params
        self.app.layout = html.Div([
            html.H1("Line Chart"),
            *[
                dcc.Graph(id={"type": "plotting-graph-data", "index": name}) for name in self.graph_data.keys()
            ],
            html.Div(style={'margin': '60px'}),
            html.Hr(),
            html.Div(style={'margin': '60px'}),
            html.Button('Save and rerun model', id='submit-matrices'),
            dcc.Tabs([
                dcc.Tab(label='Graphing Data (Initial Conditions)', children=[
                    *[
                        html.Div([
                            html.H2(f"DataTable for {name}"),
                            html.H3(f"Row names {dic['row_names']}") if dic["row_names"] is not None else None,
                            dash_table.DataTable(
                                pd.DataFrame([dic["data"]], columns=dic["column_names"]).to_dict('records') if dic["row_names"] is None else pd.DataFrame(dic["data"], columns=dic["column_names"]).to_dict('records'),
                                id={"type": 'edit-graphing-data', 'index': name},
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
                                id={"type": 'edit-non-graphing-data-vectors', 'index': name},
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
                                id={"type": 'edit-non-graphing-data-matrices', 'index': name},
                                columns=[{'name': f"{col}", 'id': col} for col in dic["column_names"]],
                                editable=True
                            ),
                        ]) for name, dic in self.non_graph_data_matrix.items()
                    ]
                ]),
                dcc.Tab(label='Environment Parameters', children=[
                    html.Div([
                        html.H2(f"DataTable for Environment Parameters"),
                        html.H4(f"Note: Some prameters wont influence the simulation. For example, changing M wont affect the number of steps in the lysis process, but overall should ahve an immediate effect on the simulation."),
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
            html.Hr(),
            html.Div(style={'margin': '60px'}),
            dcc.Tabs([
                dcc.Tab(label='Serial Transfer', children=[
                    html.H4(["Note: Using the serial transfer function will take the final iteration values of the current simulation as shown above and divide it by the value shown below. So if the final value for a bacteria is 100 and the serial transfer value is 10, the new simulation will start with a bacteria value of 10. There is a special case for nutrients where the new value is added to the value associated in the Graphing Data (Initial Conditions) section. So for nutrients, if the final value is 100 and the serial transfer value is 10, and the \"Initial\" Condition is 50, the new simulation will start with a nutrient value of 100/10 + 50 = 60. If the checkbox is selected, the unique process will also apply to the phages and bacteria. If the checkbox is not selected, the unique process will only apply to the resources/nutrients. Change the value below to 1 if oyu want to simply add phages/bacteria/resources without removing substances. "]),
                    dcc.Input(
                        id="serial_transfer_value", type="number", placeholder="input with range",
                        min=1, max=1_000_000, step=0.01, value=10
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'Add Phages and Bacteria', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='serial_tranfer_option'
                    ),
                    html.Button("Run Serial Transfer", id="run_serial_transfer"),
                    html.Div(style={'margin': '60px'}),
                ]),
                dcc.Tab(label='Parameter Analysis', children=[
                    html.H4(["Note: Choose 2 parameters of choice. Input the values you want to test separated by commas. The program will run the simulation for each combination of the two parameters and display the results in a heatmap. The heatmap represents the end value of the simulation for each combination of the two parameters. Make sure you choose an appropriate range of values to test and end simulation lenght before everything drops to 0!"]),
                    html.H4(["Choose two parameters to analyze"]),
                    dcc.Dropdown(both_params, id='parameter_analysis_dropdown_1', value = both_params[0] if len(both_params) > 0 else None),
                    dcc.Dropdown(both_params, id='parameter_analysis_dropdown_2', value = both_params[1] if len(both_params) > 1 else None),
                    dcc.Checklist(
                        options=[
                            {'label': 'Checked for running Option 1, unchecked for Option 2', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='parameter_analysis_option'
                    ),
                    html.H4(["Option 1: Input the values you want to test separated by commas"]),
                    dcc.Input(
                        id="parameter_1_input", 
                        type="text",
                        placeholder="Parameter 1 values to test",
                        value="0.01, 0.1, 1, 5"
                    ),
                    dcc.Input(
                        id="parameter_2_input", 
                        type="text",
                        placeholder="Parameter 2 values to test",
                        value="0.02, 0.2, 2, 6, 10, 20"
                    ),
                    html.Br(),
                    html.H4(["Option 2: Choose a start value and end value for each parameter separated by a '-' sign"]),
                    dcc.Input(
                        id="uniform_input_range_1",
                        type="text",
                        placeholder="0.01-0.8",
                    ), 
                    dcc.Input(
                        id="uniform_input_range_2",
                        type="text",
                        placeholder="0.01-0.8",
                    ), 
                    html.Br(),
                    html.H4(["And choose the values to test between the two values for a uniform distribution (including the end values)"]),
                    dcc.Input(
                        id="uniform_number_steps_1",
                        type="number",
                        placeholder="10",
                    ),
                    dcc.Input(
                        id="uniform_number_steps_2",
                        type="number",
                        placeholder="10",
                    ),
                    html.Br(),
                    html.Button("Run Parameter Analysis", id="run_parameter_analysis"),
                    html.Div(style={'margin': '60px'}),
                    *[
                        dcc.Graph(id={"type": "plotting-parameter-analysis-data", "index": name}) for name in self.graph_data.keys()
                    ],
                ]),
                dcc.Tab(label='Initial Value Analysis', children=[
                    html.H4(["Note: Choose a parameter of choice. Input the values you want to test separated by commas, or use a uniform seperated list. The program will run the simulation for each initial value and display the results on a graph."]),
                    dcc.Dropdown(both_params, id='initial_value_analysis_dropdown', value = both_params[0] if len(both_params) > 0 else None),
                    dcc.Checklist(
                        options=[
                            {'label': 'Checked for running Option 1, unchecked for Option 2', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='initial_value_option'
                    ),
                    html.H4(["Option 1: Input the values you want to test separated by commas"]),
                    dcc.Input(
                        id="initial_value_input", 
                        type="text",
                        placeholder="Initial values to test",
                        value="1, 10, 100, 1000"
                    ),
                    html.Br(),
                    html.H4(["Option 2: Choose a start value and end value for each parameter separated by a '-' sign"]),
                    dcc.Input(
                        id="uniform_initial_input_range",
                        type="text",
                        placeholder="1-100",
                    ),
                    html.Br(),
                    html.H4(["And choose the values to test between the two values for a uniform distribution (including the end values)"]),
                    dcc.Input(
                        id="uniform_initial_number_steps",
                        type="number",
                        placeholder="10",
                    ),
                    html.Button("Run Initial Value Analysis", id="run_initial_value_analysis"),
                    html.Div(style={'margin': '60px'}),
                    *[
                        dcc.Graph(id={"type": "plotting-initial-value-analysis-data", "index": name}) for name in self.graph_data.keys()
                    ],
                ]),
            ]),
        ])

        @callback(
            [Output({'type': 'plotting-graph-data', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()],
            Input('submit-matrices', 'n_clicks'),
            State({'type': 'edit-graphing-data', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-vectors', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-matrices', 'index': ALL}, 'data'),
            State({'type': 'environment variables', 'index': "environment variables"}, 'data'),
            prevent_initial_call=True
        )
        def rerun_matrices(n_clicks, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            _, flattened, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, graphing_data_vectors, graphing_data_matrices)
            self.graph.add_environment_data(environment_data[0])
            new_updated_data = self.graph.solve_system(self.graph.odesystem, flattened, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices)
            solved_y = new_updated_data.y
            self.copy_of_simulation_output = new_updated_data
            unflattened_data = self.graph.unflatten_initial_matrix(solved_y, [length["data"].size for length in self.graph_data.values()])
            unflattened_data = self.save_data(unflattened_data, new_updated_data.t)
            list_of_figs = self.create_figures(unflattened_data, new_updated_data.t)
            return list_of_figs
        
        @callback(
            [Output({'type': 'plotting-graph-data', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()],
            Input('run_serial_transfer', 'n_clicks'),
            State('serial_transfer_value', 'value'),
            State('serial_tranfer_option', 'value'),
            State({'type': 'edit-graphing-data', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-vectors', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-matrices', 'index': ALL}, 'data'),
            State({'type': 'environment variables', 'index': "environment variables"}, 'data'),
            prevent_initial_call=True
        )
        def serial_transfer(n_clicks, serial_transfer, serial_tranfer_option, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            _, flattened, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, graphing_data_vectors, graphing_data_matrices)
            self.graph.add_environment_data(environment_data[0])

            original_time = self.copy_of_simulation_output.t
            original_final_time = self.copy_of_simulation_output.t[-1]
            original_simulation_output = self.copy_of_simulation_output.y
            original_final_simulation_output = self.copy_of_simulation_output.y[:, -1]

            serial_transfer_flattened = self.serial_transfer_calculation(original_final_simulation_output, serial_transfer, serial_tranfer_option, flattened)
            new_updated_data = self.graph.solve_system(self.graph.odesystem, serial_transfer_flattened, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices, t_start=float(original_final_time), t_end=float(original_final_time) + float(self.graph.Simulation_Length))

            solved_y = new_updated_data.y
            new_overall_t = np.concatenate((original_time, new_updated_data.t))
            new_overall_y = np.concatenate((original_simulation_output, solved_y), axis=1)
            self.copy_of_simulation_output.t = new_overall_t
            self.copy_of_simulation_output.y = new_overall_y
            unflattened_data = self.graph.unflatten_initial_matrix(new_overall_y, [length["data"].size for length in self.graph_data.values()])

            unflattened_data = self.save_data(unflattened_data, new_updated_data.t)
            list_of_figs = self.create_figures(unflattened_data, new_updated_data.t)
            return list_of_figs
        
        @callback(
            [Output({'type': 'plotting-parameter-analysis-data', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()],
            Input('run_parameter_analysis', 'n_clicks'),
            State('parameter_analysis_dropdown_1', 'value'),
            State('parameter_analysis_dropdown_2', 'value'),
            State('parameter_analysis_option', 'value'),
            State('parameter_1_input', 'value'),
            State('parameter_2_input', 'value'),
            State('uniform_input_range_1', 'value'),
            State('uniform_input_range_2', 'value'),
            State('uniform_number_steps_1', 'value'),
            State('uniform_number_steps_2', 'value'),
            State({'type': 'edit-graphing-data', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-vectors', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-matrices', 'index': ALL}, 'data'),
            State({'type': 'environment variables', 'index': "environment variables"}, 'data'),
            prevent_initial_call=True
        )
        def parameter_analysis(n_clicks, parameter_1_name, parameter_2_name, parameter_option, parameter_1_input, parameter_2_input, uniform_range_1, uniform_range_2, uniform_steps_1, uniform_steps_2, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            _, flattened, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, graphing_data_vectors, graphing_data_matrices)
            self.graph.add_environment_data(environment_data[0])

            if len(parameter_option) > 0:
                parameter_1_values = [float(value.strip()) for value in parameter_1_input.split(",")]
                parameter_2_values = [float(value.strip()) for value in parameter_2_input.split(",")]
            else:
                start_1, end_1 = [float(value.strip()) for value in uniform_range_1.split("-")]
                start_2, end_2 = [float(value.strip()) for value in uniform_range_2.split("-")]
                parameter_1_values = np.linspace(start_1, end_1, int(uniform_steps_1)).tolist()
                parameter_2_values = np.linspace(start_2, end_2, int(uniform_steps_2)).tolist()
            matrix_output = np.zeros((len(parameter_1_values), len(parameter_2_values), len(self.graph_data)))
            items_of_name = []
            for key, value in self.graph_data.items():
                items_of_name += [key] * value["data"].size

            for parameter_1_value in parameter_1_values:
                for parameter_2_value in parameter_2_values:
                    if parameter_1_name in self.graph_data:
                        index = items_of_name.index(parameter_1_name)
                        flattened[index] = parameter_1_value
                    elif parameter_1_name in self.non_graph_data_vector:
                        new_non_graphing_data_vectors[list(self.non_graph_data_vector.keys()).index(parameter_1_name)][0] = parameter_1_value
                    else:
                        new_non_graphing_data_matrices[list(self.non_graph_data_matrix.keys()).index(parameter_1_name)][0][0] = parameter_1_value
                    if parameter_2_name in self.graph_data:
                        index = items_of_name.index(parameter_2_name)
                        flattened[index] = parameter_2_value
                    elif parameter_2_name in self.non_graph_data_vector:
                        new_non_graphing_data_vectors[list(self.non_graph_data_vector.keys()).index(parameter_2_name)][0] = parameter_2_value
                    elif parameter_2_name in self.non_graph_data_matrix:
                        new_non_graphing_data_matrices[list(self.non_graph_data_matrix.keys()).index(parameter_2_name)][0][0] = parameter_2_value

                    new_updated_data = self.graph.solve_system(self.graph.odesystem, flattened, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices)
                    solved_y = new_updated_data.y
                    unflattened_data = self.graph.unflatten_initial_matrix(solved_y, [length["data"].size for length in self.graph_data.values()])
                    unflattened_data = self.save_data(unflattened_data, new_updated_data.t)
                    for i, data in enumerate(unflattened_data):
                        matrix_output[parameter_1_values.index(parameter_1_value), parameter_2_values.index(parameter_2_value), i] = data[0][-1]
            list_of_fig_heatmaps = []
            for i, name in zip(range(matrix_output.shape[2]), self.graph_data.keys()):
                list_of_fig_heatmaps.append(self.create_heatmap(matrix_output[:, :, i], parameter_1_values, parameter_2_values, parameter_1_name, parameter_2_name, f"Parameter {parameter_1_name} vs {parameter_2_name} Analysis for {name}"))
            return list_of_fig_heatmaps

        @callback(
            [Output({'type': 'plotting-initial-value-analysis-data', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()],
            Input('run_initial_value_analysis', 'n_clicks'),
            State('initial_value_analysis_dropdown', 'value'),
            State('initial_value_option', 'value'),
            State('initial_value_input', 'value'),
            State('uniform_initial_input_range', 'value'),
            State('uniform_initial_number_steps', 'value'),
            State({'type': 'edit-graphing-data', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-vectors', 'index': ALL}, 'data'),
            State({'type': 'edit-non-graphing-data-matrices', 'index': ALL}, 'data'),
            State({'type': 'environment variables', 'index': "environment variables"}, 'data'),
            prevent_initial_call=True
        )
        def starting_analysis(n_clicks, initial_name, initial_option, initial_input, initial_input_range, initial_number_steps, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            _, flattened, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, graphing_data_vectors, graphing_data_matrices)
            self.graph.add_environment_data(environment_data[0])

            if len(initial_option) > 0:
                parameter_1_values = [float(value.strip()) for value in initial_input.split(",")]
            else:
                start_1, end_1 = [float(value.strip()) for value in initial_input_range.split("-")]
                parameter_1_values = np.linspace(start_1, end_1, int(initial_number_steps)).tolist()
            items_of_name = []
            for key, value in self.graph_data.items():
                items_of_name += [key] * value["data"].size
            simulation_output = []
            time_output = []
            for parameter_1_value in parameter_1_values:
                if initial_name in self.graph_data:
                    index = items_of_name.index(initial_name)
                    flattened[index] = parameter_1_value
                elif initial_name in self.non_graph_data_vector:
                    new_non_graphing_data_vectors[list(self.non_graph_data_vector.keys()).index(initial_name)][0] = parameter_1_value
                elif initial_name in self.non_graph_data_matrix:
                    new_non_graphing_data_matrices[list(self.non_graph_data_matrix.keys()).index(initial_name)][0] = parameter_1_value
                new_updated_data = self.graph.solve_system(self.graph.odesystem, flattened, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices)
                solved_y = new_updated_data.y
                unflattened_data = self.graph.unflatten_initial_matrix(solved_y, [length["data"].size for length in self.graph_data.values()])
                unflattened_data = self.save_data(unflattened_data, new_updated_data.t)
                simulation_output.append(unflattened_data)
                time_output.append(new_updated_data.t)
            list_of_figs = []
            for i, name in zip(range(len(self.graph_data.keys())), self.graph_data.keys()):
                fig = go.Figure(dict(text=name))
                for j in range(len(simulation_output)):
                    fig.add_trace(go.Scatter(x=time_output[j], y=simulation_output[j][i][0], mode="lines", name=f"{initial_name} {parameter_1_values[j]}"))
                    fig.update_layout(
                        title=f"Initial Value Analysis for {name}",
                        xaxis=dict(title="Time"),
                        yaxis=dict(title="Value")
                    )
                list_of_figs.append(fig)
            return list_of_figs

        self.app.run_server(debug=True)