import numpy as np
from Classes.Analysis import Analysis
from Classes.GraphMakerGUI import GraphMakerGUI
import plotly.express as px
import pandas as pd
from dash import Dash, dash_table, html, Input, Output, callback, ALL, State
from dash import dcc
import plotly.graph_objs as go
from collections import OrderedDict
import plotly.figure_factory as ff
from Classes.VisualizerHTML import html_code

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
        return row_of_values
    
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
        self.app.layout = html_code(self.graph_data, self.non_graph_data_vector, self.non_graph_data_matrix, self.graph)

        @callback(
            [Output({'type': 'plot_basic_graph_data', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()],
            Input('run_basic_model', 'n_clicks'),
            State({'type': 'edit_graphing_data', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_vectors', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_matrices', 'index': ALL}, 'data'),
            State('environment_data', 'data'),
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
            [Output({'type': 'plot_basic_graph_data', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()],
            Input('run_serial_transfer', 'n_clicks'),
            State('serial_transfer_value', 'value'),
            State('serial_tranfer_bp_option', 'value'),
            State('serial_transfer_frequency', 'value'),
            State({'type': 'edit_graphing_data', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_vectors', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_matrices', 'index': ALL}, 'data'),
            State('environment_data', 'data'),
            prevent_initial_call=True
        )
        def serial_transfer(n_clicks, serial_transfer_value, serial_tranfer_bp_option, serial_transfer_frequency, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            print(graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data)
            _, flattened, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, graphing_data_vectors, graphing_data_matrices)
            self.graph.add_environment_data(environment_data[0])
            new_overall_t = self.copy_of_simulation_output.t
            new_overall_y = self.copy_of_simulation_output.y
            for _ in range(int(serial_transfer_frequency)):
                original_time = new_overall_t
                original_final_time = new_overall_t[-1]
                original_simulation_output = new_overall_y
                original_final_simulation_output = new_overall_y[:, -1]

                serial_transfer_flattened = self.serial_transfer_calculation(original_final_simulation_output, serial_transfer_value, serial_tranfer_bp_option, flattened)
                new_updated_data = self.graph.solve_system(self.graph.odesystem, serial_transfer_flattened, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices, t_start=float(original_final_time), t_end=float(original_final_time) + float(self.graph.Simulation_Length))

                solved_y = new_updated_data.y
                new_overall_t = np.concatenate((original_time, new_updated_data.t))
                new_overall_y = np.concatenate((original_simulation_output, solved_y), axis=1)
                unflattened_data = self.graph.unflatten_initial_matrix(new_overall_y, [length["data"].size for length in self.graph_data.values()])
                unflattened_data = self.save_data(unflattened_data, new_overall_t, save_data=False)
            self.copy_of_simulation_output.t = new_overall_t
            self.copy_of_simulation_output.y = new_overall_y
            list_of_figs = self.create_figures(unflattened_data, new_overall_t)
            return list_of_figs
        
        @callback(
            [Output({'type': 'plot_parameter_analysis', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()],
            Input('run_parameter_analysis', 'n_clicks'),
            State('parameter_analysis_param_name_1', 'value'),
            State('parameter_analysis_param_name_2', 'value'),
            State('parameter_analysis_option', 'value'),
            State('parameter_analysis_input_1', 'value'),
            State('parameter_analysis_input_2', 'value'),
            State('parameter_analysis_range_1', 'value'),
            State('parameter_analysis_range_2', 'value'),
            State('parameter_analysis_steps_1', 'value'),
            State('parameter_analysis_steps_2', 'value'),
            State('parameter_analysis_use_serial_transfer', 'value'),
            State('serial_transfer_value', 'value'),
            State('serial_tranfer_bp_option', 'value'),
            State('serial_transfer_frequency', 'value'),
            State({'type': 'edit_graphing_data', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_vectors', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_matrices', 'index': ALL}, 'data'),
            State('environment_data', 'data'),
            prevent_initial_call=True
        )
        def parameter_analysis(n_clicks, param_1_name, param_2_name, use_opt_1_or_opt_2, param_1_input, param_2_input, param_range_1, param_range_2, param_steps_1, param_steps_2, use_serial_transfer, serial_transfer_value, serial_transfer_bp_option, serial_transfer_frequency, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            _, flattened, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, graphing_data_vectors, graphing_data_matrices)
            self.graph.add_environment_data(environment_data[0])

            if use_opt_1_or_opt_2:
                parameter_1_values = [float(value.strip()) for value in param_1_input.split(",")]
                parameter_2_values = [float(value.strip()) for value in param_2_input.split(",")]
            else:
                start_1, end_1 = [float(value.strip()) for value in param_range_1.split("-")]
                start_2, end_2 = [float(value.strip()) for value in param_range_2.split("-")]
                parameter_1_values = np.linspace(start_1, end_1, int(param_steps_1)).tolist()
                parameter_2_values = np.linspace(start_2, end_2, int(param_steps_2)).tolist()
            matrix_output = np.zeros((len(parameter_1_values), len(parameter_2_values), len(self.graph_data)))
            items_of_name = []
            for key, value in self.graph_data.items():
                items_of_name += [key] * value["data"].size

            for parameter_1_value in parameter_1_values:
                for parameter_2_value in parameter_2_values:
                    if param_1_name in self.graph_data:
                        index = items_of_name.index(param_1_name)
                        flattened[index] = parameter_1_value
                    elif param_1_name in self.non_graph_data_vector:
                        new_non_graphing_data_vectors[list(self.non_graph_data_vector.keys()).index(param_1_name)][0] = parameter_1_value
                    else:
                        new_non_graphing_data_matrices[list(self.non_graph_data_matrix.keys()).index(param_1_name)][0][0] = parameter_1_value
                    if param_2_name in self.graph_data:
                        index = items_of_name.index(param_2_name)
                        flattened[index] = parameter_2_value
                    elif param_2_name in self.non_graph_data_vector:
                        new_non_graphing_data_vectors[list(self.non_graph_data_vector.keys()).index(param_2_name)][0] = parameter_2_value
                    elif param_2_name in self.non_graph_data_matrix:
                        new_non_graphing_data_matrices[list(self.non_graph_data_matrix.keys()).index(param_2_name)][0][0] = parameter_2_value
                    
                    new_updated_data = self.graph.solve_system(self.graph.odesystem, flattened, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices)
                    solved_y = new_updated_data.y
                    solved_t = new_updated_data.t
                    last_values = solved_y[:, -1]
                    if use_serial_transfer:
                        for _ in range(int(serial_transfer_frequency)):
                            flattened_copy = flattened.copy()
                            flattened_copy = self.serial_transfer_calculation(last_values, serial_transfer_value, serial_transfer_bp_option, flattened_copy)
                            new_updated_data = self.graph.solve_system(self.graph.odesystem, flattened_copy, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices)
                            solved_y = np.concatenate((solved_y, new_updated_data.y), axis=1)
                            solved_t = np.concatenate((solved_t, new_updated_data.t))
                            last_values = new_updated_data.y[:, -1]
                    unflattened_data = self.graph.unflatten_initial_matrix(solved_y, [length["data"].size for length in self.graph_data.values()])
                    unflattened_data = self.save_data(unflattened_data, solved_t, save_data=False)
                    for i, data in enumerate(unflattened_data):
                        matrix_output[parameter_1_values.index(parameter_1_value), parameter_2_values.index(parameter_2_value), i] = data[0][-1]
            list_of_fig_heatmaps = []
            for i, name in zip(range(matrix_output.shape[2]), self.graph_data.keys()):
                list_of_fig_heatmaps.append(self.create_heatmap(matrix_output[:, :, i], parameter_1_values, parameter_2_values, param_1_name, param_2_name, f"Parameter {param_1_name} vs {param_2_name} Analysis for {name}"))
            return list_of_fig_heatmaps

        @callback(
            [Output({'type': 'plot_initial_value_analysis', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()],
            Input('run_initial_value_analysis', 'n_clicks'),
            State('initial_value_analysis_param_name', 'value'),
            State('initial_value_analysis_option', 'value'),
            State('initial_value_analysis_input', 'value'),
            State('initial_value_analysis_range', 'value'),
            State('initial_value_analysis_steps', 'value'),
            State('initial_value_analysis_use_serial_transfer', 'value'),
            State('serial_transfer_value', 'value'),
            State('serial_tranfer_bp_option', 'value'),
            State('serial_transfer_frequency', 'value'),
            State({'type': 'edit_graphing_data', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_vectors', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_matrices', 'index': ALL}, 'data'),
            State('environment_data', 'data'),
            prevent_initial_call=True
        )
        def initial_value_analysis(n_clicks, param_name, use_opt_1_or_opt_2, param_input, param_range, param_steps, use_serial_transfer, serial_transfer_value, serial_transfer_bp_option, serial_transfer_frequency, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            _, flattened, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, graphing_data_vectors, graphing_data_matrices)
            self.graph.add_environment_data(environment_data[0])

            if len(use_opt_1_or_opt_2) > 0:
                parameter_1_values = [float(value.strip()) for value in param_input.split(",")]
            else:
                start_1, end_1 = [float(value.strip()) for value in param_range.split("-")]
                parameter_1_values = np.linspace(start_1, end_1, int(param_steps)).tolist()
            items_of_name = []
            for key, value in self.graph_data.items():
                items_of_name += [key] * value["data"].size
            simulation_output = []
            time_output = []
            for parameter_1_value in parameter_1_values:
                if param_name in self.graph_data:
                    index = items_of_name.index(param_name)
                    flattened[index] = parameter_1_value
                elif param_name in self.non_graph_data_vector:
                    new_non_graphing_data_vectors[list(self.non_graph_data_vector.keys()).index(param_name)][0] = parameter_1_value
                elif param_name in self.non_graph_data_matrix:
                    new_non_graphing_data_matrices[list(self.non_graph_data_matrix.keys()).index(param_name)][0] = parameter_1_value
                new_updated_data = self.graph.solve_system(self.graph.odesystem, flattened, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices)
                solved_y = new_updated_data.y
                solved_t = new_updated_data.t
                unflattened_data = self.graph.unflatten_initial_matrix(solved_y, [length["data"].size for length in self.graph_data.values()])
                last_values = solved_y[:, -1]
                if use_serial_transfer:
                    for _ in range(int(serial_transfer_frequency)):
                        flattened_copy = flattened.copy()
                        flattened_copy = self.serial_transfer_calculation(last_values, serial_transfer_value, serial_transfer_bp_option, flattened_copy)
                        new_updated_data = self.graph.solve_system(self.graph.odesystem, flattened_copy, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices, t_start=float(solved_t[-1]), t_end=float(solved_t[-1]) + float(self.graph.Simulation_Length))
                        solved_y = np.concatenate((solved_y, new_updated_data.y), axis=1)
                        solved_t = np.concatenate((solved_t, new_updated_data.t))
                        last_values = new_updated_data.y[:, -1]
                unflattened_data = self.graph.unflatten_initial_matrix(solved_y, [length["data"].size for length in self.graph_data.values()])
                unflattened_data = self.save_data(unflattened_data, solved_t, save_data=False)
                simulation_output.append(unflattened_data)
                time_output.append(solved_t)
            list_of_figs = []
            for i, name in zip(range(len(self.graph_data.keys())), self.graph_data.keys()):
                fig = go.Figure(dict(text=name))
                for j in range(len(simulation_output)):
                    fig.add_trace(go.Scatter(x=time_output[j], y=simulation_output[j][i][0], mode="lines", name=f"{param_name} {parameter_1_values[j]}"))
                    fig.update_layout(
                        title=f"Initial Value Analysis for {name}",
                        xaxis=dict(title="Time"),
                        yaxis=dict(title="Value")
                    )
                list_of_figs.append(fig)
            return list_of_figs
        
        @callback(
            Output('plot_phase_portrait', 'figure', allow_duplicate=True),
            Input('run_phase_portrait', 'n_clicks'),
            State('phase_portrait_param_name_1', 'value'),
            State('phase_portrait_param_name_2', 'value'),
            State('phase_portrait_range_1', 'value'),
            State('phase_portrait_steps_1', 'value'),
            State('phase_portrait_range_2', 'value'),
            State('phase_portrait_steps_2', 'value'),
            State('phase_portrait_use_serial_transfer', 'value'),
            State('serial_transfer_value', 'value'),
            State('serial_tranfer_bp_option', 'value'),
            State('serial_transfer_frequency', 'value'),
            State({'type': 'edit_graphing_data', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_vectors', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_matrices', 'index': ALL}, 'data'),
            State('environment_data', 'data'),
            prevent_initial_call=True
        )
        def phase_portrait(n_clicks, param_1_name, param_2_name, param_range_1, param_steps_1, param_range_2, param_steps_2, use_serial_transfer, serial_transfer_value, serial_transfer_bp_option, serial_transfer_frequency, graphing_data, graphing_data_vectors, graphing_data_matrices, environment_data):
            _, flattened, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, graphing_data_vectors, graphing_data_matrices)
            self.graph.add_environment_data(environment_data[0])

            input_value_1_low, input_value_1_high = param_range_1.split("-")
            input_value_2_low, input_value_2_high = param_range_2.split("-")
            x_vals = np.linspace(float(input_value_1_low), float(input_value_1_high), int(param_steps_1))
            y_vals = np.linspace(float(input_value_2_low), float(input_value_2_high), int(param_steps_1))
            X, Y = np.meshgrid(x_vals, y_vals)

            DX, DY = np.zeros(X.shape), np.zeros(Y.shape)
            items_of_name_1 = []
            items_of_name_2 = []
            for key, value in self.graph_data.items():
                items_of_name_1 += [key] * value["data"].size
                items_of_name_2 += [key]
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    flattened[items_of_name_1.index(param_1_name)] = X[i, j]
                    flattened[items_of_name_1.index(param_2_name)] = Y[i, j]
                    new_updated_data = self.graph.odesystem(0, flattened, *[self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices])
                    value1 = new_updated_data[items_of_name_1.index(param_1_name)]
                    value2 = new_updated_data[items_of_name_1.index(param_2_name)]
                    DX[i, j], DY[i, j] = value1, value2

            # Normalize arrows for better visualization
            M = np.hypot(DX, DY)
            DX, DY = DX / M, DY / M  # Normalize to unit vectors
            fig = ff.create_quiver(X, Y, DX, DY, 
                scale=0.3,
                arrow_scale=0.3,
                name='quiver',
                line_width=3, 
                angle=0.1,
                hovertemplate=f"{param_1_name}: %{{x}}<br>{param_2_name}: %{{y}}", 
            )
            fig.update_layout(
                title=f"Phase Portrait for {param_1_name} vs {param_2_name}",
                xaxis=dict(title=param_1_name),
                yaxis=dict(title=param_2_name),
                width=1200, 
                height=800
            )
            _, flattened, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, graphing_data_vectors, graphing_data_matrices)
            self.graph.add_environment_data(environment_data[0])
            new_updated_data = self.graph.solve_system(self.graph.odesystem, flattened, self.graph, *self.other_parameters_to_pass, *new_non_graphing_data_vectors, *new_non_graphing_data_matrices)
            solved_y = new_updated_data.y
            self.copy_of_simulation_output = new_updated_data
            unflattened_data = self.graph.unflatten_initial_matrix(solved_y, [length["data"].size for length in self.graph_data.values()])
            unflattened_data = self.save_data(unflattened_data, new_updated_data.t)
            value1 = unflattened_data[items_of_name_1.index(param_1_name)][0]
            value2 = unflattened_data[items_of_name_2.index(param_2_name)][0]
            fig.add_trace(
                go.Scatter(x=value1, y=value2, mode="lines", name=f"{param_1_name} vs {param_2_name}", hovertemplate=f"{param_1_name}: %{{x}}<br>{param_2_name}: %{{y}}<br>time: %{{meta}}<extra></extra>",
                meta=new_updated_data.t)
            )
            return fig

        self.app.run_server(debug=True)