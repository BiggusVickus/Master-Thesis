import numpy as np
from Classes.Analysis import Analysis
import plotly.express as px
import pandas as pd
from dash import Dash, Input, Output, callback, ALL, State
import plotly.graph_objs as go
from collections import OrderedDict
import plotly.figure_factory as ff
from Classes.VisualizerHTML import html_code, parse_contents
from plotly.subplots import make_subplots
#TODO: give option to append a serial transfer to parameter analysis and initial value analysis
class Visualizer():
    def __init__(self, graph):
        self.app = Dash()
        self.graph: Analysis = graph
        self.graph_data = OrderedDict()
        self.non_graph_data_vector = OrderedDict()
        self.non_graph_data_matrix = OrderedDict()
        self.other_parameters_to_pass = []
        self.copy_of_simulation_output = None
        self.copy_of_parameter_analysis_output = None
        self.settings = {}

    def add_graph_data(self, name, data, column_names, row_names=None, add_rows=False):
        self.graph_data[name] = {"data": data, "column_names": column_names, "row_names": row_names, "add_rows": add_rows}

    def add_non_graph_data_vector(self, name, data, column_names):
        self.non_graph_data_vector[name] = {"data": data, "column_names": column_names}

    def add_non_graph_data_matrix(self, name, data, row_names, column_names):
        self.non_graph_data_matrix[name] = {"data": data, "row_names": row_names, "column_names": column_names}

    def add_other_parameters(self, *args):
        self.other_parameters_to_pass += args
    
    def create_main_figures(self, unflattened_data, overall_t):
        list_of_figs = []
        for i, dictionary in enumerate(self.graph_data.items()):
            name, dic = dictionary
            fig = go.Figure(dict(text=name))
            for j in range(len(unflattened_data[i])):
                fig.add_trace(go.Scatter(x=overall_t, y=unflattened_data[i][j], mode="lines", name=f"{dic['column_names'][j]}"))
                fig.update_layout(
                    title=f"Graph for {name}",
                    xaxis=dict(title="Time"),
                    yaxis=dict(title="Value")
                )
            list_of_figs.append(fig)
        return list_of_figs
    
    def create_initial_value_analysis_figures(self, simulation_output, time_output, param_name, param_values):
        list_of_figs = []
        for i, name in zip(range(len(self.graph_data.keys())), self.graph_data.keys()):
            fig = go.Figure(dict(text=name))
            for j in range(len(simulation_output)):
                fig.add_trace(go.Scatter(x=time_output[j], y=simulation_output[j][i][0], mode="lines", name=f"{param_name} {param_values[j]}"))
                fig.update_layout(
                    title=f"Initial Value Analysis for {name}",
                    xaxis=dict(title="Time"),
                    yaxis=dict(title="Value")
                )
            list_of_figs.append(fig)
        return list_of_figs
    
    def create_numpy_lists(self, graphing_data, graphing_data_vectors, graphing_data_matrices):
        graphing_data = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data]
        flattened = self.graph.flatten_lists_and_matrices(*graphing_data)
        non_graphing_data_vectors = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy()[0] for data_values in graphing_data_vectors]
        non_graphing_data_matrices = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data_matrices]
        return graphing_data, flattened, non_graphing_data_vectors, non_graphing_data_matrices
    
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
        
    def save_data(self, array, time, save_data=True):
        unflattened_data = []
        for dic, data_item in zip(self.graph_data.items(), array):
            key, value = dic
            if save_data:
                self.graph_data[key]["y_data"] = data_item
                self.graph_data[key]["t_data"] = time
            unflattened_data.append(self.sum_up_columns(data_item, value["add_rows"]))
        return unflattened_data
    
    def create_heatmap_figures(self, matrix_data, x_axis_data=None, y_axis_data=None, x_labels=None, y_labels=None):
        list_of_figs = []
        for i, name in zip(range(matrix_data.shape[2]), self.graph_data.keys()):
            df = pd.DataFrame(matrix_data[:, :, i], columns=y_axis_data, index=x_axis_data)
            fig = px.imshow(
                df, 
                labels={'x': y_labels, 'y': x_labels}, 
                text_auto=True, 
                aspect="equal" 
            )
            fig.update_layout(
                title=f"Parameter {x_labels} vs {y_labels} Analysis for {name}",
                uirevision='constant',
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(y_axis_data))), 
                    ticktext=y_axis_data, 
                    categoryarray=y_axis_data,      
                ),
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(x_axis_data))), 
                    ticktext=x_axis_data,
                    categoryarray=x_axis_data,
                ), 
                autosize=False,
                width=1000,
                height=1000
            )
            fig.update_traces(
                hovertemplate=f"{y_labels}: %{{x}}<br>{x_labels}: %{{y}}<br>End Value: %{{z}}<extra></extra>"
            )
            fig.update_xaxes(type='category')
            fig.update_yaxes(type='category')
            list_of_figs.append(fig)
        return list_of_figs
        
    def run_serial_transfer_iterations(self, overall_y, overall_t, serial_transfer_frequency, flattened, serial_transfer_value, serial_transfer_bp_option, non_graphing_data_vectors, non_graphing_data_matrices):
        final_values = overall_y[:, -1]
        final_time = overall_t[-1]
        for _ in range(int(serial_transfer_frequency)):
            flattened = self.serial_transfer_calculation(final_values, serial_transfer_value, serial_transfer_bp_option, flattened)
            solved_system = self.graph.solve_system(self.graph.odesystem, flattened, self.graph, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices, t_start=float(final_time), t_end=float(final_time) + float(self.graph.Simulation_Length))
            overall_y = np.concatenate((overall_y, solved_system.y), axis=1)
            overall_t = np.concatenate((overall_t, solved_system.t))
            final_values = solved_system.y[:, -1]
            final_time = solved_system.t[-1]
        return overall_y, overall_t
    
    def split_comma_minus(self, input, range, steps, use_opt_1_or_opt_2):
        if use_opt_1_or_opt_2:
            return [float(value.strip()) for value in input.split(",")]
        else:
            start_1, end_1 = [float(value.strip()) for value in range.split("-")]
            return np.linspace(start_1, end_1, int(steps)).tolist()
    
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
        def plot_main_plots(n_clicks, graphing_data, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
            # turn the data in the dashboard into numpy arrays, and save/update the environemnt data to the graph object
            _, flattened, non_graphing_data_vectors, non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.graph.update_environment_data(environment_data[0])

            # solves sytem of ODEs, saves y and t data results
            solved_system = self.graph.solve_system(self.graph.odesystem, flattened, self.graph, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices)
            self.copy_of_simulation_output = solved_system

            # unflattens the data, saves the data to the graph_data dictionary, and sums up the columns if necessary
            overall_y = self.graph.unflatten_initial_matrix(solved_system.y, [length["data"].size for length in self.graph_data.values()])
            overall_y = self.save_data(overall_y, solved_system.t)
            list_of_figs = self.create_main_figures(overall_y, solved_system.t)
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
        def serial_transfer(n_clicks, serial_transfer_value, serial_tranfer_bp_option, serial_transfer_frequency, graphing_data, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
            # turn the data in the dashboard into numpy arrays, and save/update the environemnt data to the graph object
            _, initial_condition, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.graph.update_environment_data(environment_data[0])

            #use previously saved data to start the simulation from the last time point
            overall_t = self.copy_of_simulation_output.t
            overall_y = self.copy_of_simulation_output.y
            
            # for the required number of runs of serial transfer
            overall_y, overall_t = self.run_serial_transfer_iterations(overall_y, overall_t, serial_transfer_frequency, initial_condition, serial_transfer_value, serial_tranfer_bp_option, new_non_graphing_data_vectors, new_non_graphing_data_matrices)
            
            # save the values to self.copy_of_simulation_output.y/t respectively, in case for future serial transfers
            self.copy_of_simulation_output.t = overall_t
            self.copy_of_simulation_output.y = overall_y

            # unflatten the data, save the data to the graph_data dictionary, and sum up the columns if necessary, then create the figures
            overall_y = self.graph.unflatten_initial_matrix(overall_y, [length["data"].size for length in self.graph_data.values()])
            overall_y = self.save_data(overall_y, overall_t, save_data=False)
            list_of_figs = self.create_main_figures(overall_y, overall_t)
            return list_of_figs
        
        @callback(
            [Output({'type': 'plot_parameter_analysis', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()], 
            Output('parameter_analysis_slider', 'max'), 
            Output('parameter_analysis_slider', 'marks'),
            Output('parameter_analysis_slider', 'value'),
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
        def parameter_analysis(n_clicks, param_name_1, param_name_2, use_opt_1_or_opt_2, param_1_input, param_2_input, param_range_1, param_range_2, param_steps_1, param_steps_2, use_serial_transfer, serial_transfer_value, serial_transfer_bp_option, serial_transfer_frequency, graphing_data, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
            # turn the data in the dashboard into numpy arrays, and save/update the environemnt data to the graph object
            _, initial_condition, non_graphing_data_vectors, non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.graph.update_environment_data(environment_data[0])

            #if option 1 is selected, then the values used to test the simulation are split by commas, and are put into a list as a float. Otherwise the range is split by a dash and linspace is used to create the values, and put into a list as a float
            param_values_1 = self.split_comma_minus(param_1_input, param_range_1, param_steps_1, use_opt_1_or_opt_2)
            param_values_2 = self.split_comma_minus(param_2_input, param_range_2, param_steps_2, use_opt_1_or_opt_2)

            # create a matrix to store the values of the final time point for each parameter value tested
            matrix_output = np.zeros((len(param_values_1), len(param_values_2), len(self.graph_data)))
            # create a list of the names of the parameters, to be used to find the index of the parameter in the flattened array
            items_of_name = []
            for key, value in self.graph_data.items():
                items_of_name += [key] * value["data"].size
            
            y_values_to_save = OrderedDict()
            t_values_to_save = OrderedDict()

            # loop through each parameter value, and solve the system of ODEs, and save the final time point value for each parameter value
            for param_value_1 in param_values_1:
                for param_value_2 in param_values_2:
                    # if the parameter 1 is in the graph data, then the index is found, and the value is set to the parameter value. Otherwise, the value is set to the parameter value in the non graph data vector or matrix
                    if param_name_1 in self.graph_data:
                        index = items_of_name.index(param_name_1)
                        initial_condition[index] = param_value_1
                    elif param_name_1 in self.non_graph_data_vector:
                        non_graphing_data_vectors[list(self.non_graph_data_vector.keys()).index(param_name_1)][0] = param_value_1
                    else:
                        non_graphing_data_matrices[list(self.non_graph_data_matrix.keys()).index(param_name_1)][0][0] = param_value_1
                    
                    # if the parameter 2 is in the graph data, then the index is found, and the value is set to the parameter value. Otherwise, the value is set to the parameter value in the non graph data vector or matrix
                    if param_name_2 in self.graph_data:
                        index = items_of_name.index(param_name_2)
                        initial_condition[index] = param_value_2
                    elif param_name_2 in self.non_graph_data_vector:
                        non_graphing_data_vectors[list(self.non_graph_data_vector.keys()).index(param_name_2)][0] = param_value_2
                    elif param_name_2 in self.non_graph_data_matrix:
                        non_graphing_data_matrices[list(self.non_graph_data_matrix.keys()).index(param_name_2)][0][0] = param_value_2
                    
                    # solve the system of ODEs, and save the final value and time value
                    solved_system = self.graph.solve_system(self.graph.odesystem, initial_condition, self.graph, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices)
                    overall_y = solved_system.y
                    overall_t = solved_system.t
                    
                    # if serial transfer is selected, then the system is run for the number of iterations specified, and the final values are saved
                    if use_serial_transfer:
                        overall_y, overall_t = self.run_serial_transfer_iterations(overall_y, overall_t, serial_transfer_frequency, initial_condition, serial_transfer_value, serial_transfer_bp_option, non_graphing_data_vectors, non_graphing_data_matrices)
                    overall_y = self.graph.unflatten_initial_matrix(overall_y, [length["data"].size for length in self.graph_data.values()])
                    overall_y = self.save_data(overall_y, overall_t, save_data=False)
                    y_values_to_save[(param_value_1, param_value_2)] = overall_y
                    t_values_to_save[(param_value_1, param_value_2)] = overall_t
                    # save the final value to the matrix
                    for i, data in enumerate(overall_y):
                        matrix_output[param_values_1.index(param_value_1), param_values_2.index(param_value_2), i] = data[0][-1]
            self.copy_of_parameter_analysis_output = {"overall_y": y_values_to_save, "overall_t": t_values_to_save, "x_axis_data": param_values_1, "y_axis_data": param_values_2, "x_labels": param_name_1, "y_labels": param_name_2}
            # Update slider value range to new values 0 to overall_t[-1]
            slider_marks = {i: f"{i:.2f}" for i in np.linspace(0, overall_t[-1], 40)}
            # create a list of figures, where each figure is a heatmap of the final values for each parameter value
            return *self.create_heatmap_figures(matrix_output, x_axis_data=param_values_1, y_axis_data=param_values_2, x_labels=param_name_1, y_labels=param_name_2), overall_t[-1], slider_marks, overall_t[-1]

        @callback(
            [Output({'type': 'plot_parameter_analysis', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()],
            # Input('parameter_analysis_slider', 'drag_value'),
            Input('parameter_analysis_slider', 'value'),
            Input('parameter_analysis_extrapolate', 'value'),
            prevent_initial_call=True
        )
        def parameter_analysis_slider_update(slider_value, extrapolate):
            # when first launching the app and opening the Parameter Analysis tab, error is thrown by dash for self.copy_of_parameter_analysis_output being None/empty, so return empty figures to avoid/fix/alleviate this error
            # collect all the stored data from the parameter analysis, and create a heatmap of the final values for each parameter value
            try:
                param_values_1 = self.copy_of_parameter_analysis_output["x_axis_data"]
            except:
                return [go.Figure() for _ in self.graph_data.keys()]
            param_values_2 = self.copy_of_parameter_analysis_output["y_axis_data"]
            param_name_1 = self.copy_of_parameter_analysis_output["x_labels"]
            param_name_2 = self.copy_of_parameter_analysis_output["y_labels"]
            overall_t = self.copy_of_parameter_analysis_output["overall_t"]
            overall_y = self.copy_of_parameter_analysis_output["overall_y"]

            # create a matrix to store the values of the final time point for each parameter value tested
            matrix_output = np.zeros((len(param_values_1), len(param_values_2), len(self.graph_data)))
            
            # loop through each set of parameter values, for the x, y point in the parameter analysis, and find the value at the time point closest to the slider value
            for param_value_1 in param_values_1:
                for param_value_2 in param_values_2:
                    # get the data for the parameter values from the overall_y and overall_t dictionaries for the parameter values
                    temp_y = overall_y[(param_value_1, param_value_2)]
                    temp_t = overall_t[(param_value_1, param_value_2)]
                    # loop thorugh each parameter value, and find the value at the time point closest to the slider value
                    for i, data in enumerate(temp_y):
                        # if extrapolate is selected, then the value is found by linearly interpolating between the two closest time points. Otherwise, the value is found by finding the left closest time point
                        if (extrapolate):
                            index_1 = np.searchsorted(temp_t, slider_value, side="left") - 1
                            index_2 = index_1 + 1
                            time_1, time_2 = temp_t[index_1], temp_t[index_2]
                            value_1, value_2 = data[0][index_1], data[0][index_2]
                            value = value_1 + (value_2 - value_1) * (slider_value - time_1) / (time_2 - time_1)
                        else:
                            # if the slider value is in the time points, then the index is found, and the value is set to the parameter value. Otherwise, the value is set to the left closest time point
                            if slider_value in temp_t:
                                index_1 = np.where(temp_t == slider_value)[0][0]
                            else:
                                index_1 = np.searchsorted(temp_t, slider_value, side="left") - 1
                            value = data[0][index_1]
                        # save the value to the matrix
                        matrix_output[param_values_1.index(param_value_1), param_values_2.index(param_value_2), i] = value

            # create a list of figures, where each figure is a heatmap of the final values for each parameter value
            return self.create_heatmap_figures(matrix_output, x_axis_data=param_values_1, y_axis_data=param_values_2, x_labels=param_name_1, y_labels=param_name_2)
        # 0.07222012281417847 for drag_value + clicking @ 100 clicks
        # 0.12108221735273089 for drag_value + dragging @ 140 drags
        # 0.06944610595703125s for vlaue + clicking @ 100 clicks

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
        def initial_value_analysis(n_clicks, param_name, use_opt_1_or_opt_2, param_input, param_range, param_steps, use_serial_transfer, serial_transfer_value, serial_transfer_bp_option, serial_transfer_frequency, graphing_data, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
            # turn the data in the dashboard into numpy arrays, and save/update the environemnt data to the graph object
            _, initial_condition, non_graphing_data_vectors, non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.graph.update_environment_data(environment_data[0])

            # if option 1 is selected, then the values used to test the simulation are split by commas, and are put into a list as a float. Otherwise the range is split by a dash and linspace is used to create the values, and put into a list as a float
            param_1_values = self.split_comma_minus(param_input, param_range, param_steps, use_opt_1_or_opt_2)
            # create a list of the names of the parameters, to be used to find the index of the parameter in the flattened array
            items_of_name = []
            for key, value in self.graph_data.items():
                items_of_name += [key] * value["data"].size

            # create a list of the simulation output, and the time output
            simulation_output = []
            time_output = []
            # loop through each parameter value, and solve the system of ODEs, and save the final time point value for each parameter value
            for param_1_value in param_1_values:
                # if the parameter 1 is in the graph data, then the index is found, and the value is set to the parameter value. Otherwise, the value is set to the parameter value in the non graph data vector or matrix
                if param_name in self.graph_data:
                    index = items_of_name.index(param_name)
                    initial_condition[index] = param_1_value
                elif param_name in self.non_graph_data_vector:
                    non_graphing_data_vectors[list(self.non_graph_data_vector.keys()).index(param_name)][0] = param_1_value
                elif param_name in self.non_graph_data_matrix:
                    non_graphing_data_matrices[list(self.non_graph_data_matrix.keys()).index(param_name)][0] = param_1_value
                # solve the system of ODEs, and save the final value and time value
                updated_data = self.graph.solve_system(self.graph.odesystem, initial_condition, self.graph, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices)
                overall_y = updated_data.y
                overall_t = updated_data.t
                if use_serial_transfer:
                    overall_y, overall_t = self.run_serial_transfer_iterations(overall_y, overall_t, serial_transfer_frequency, initial_condition, serial_transfer_value, serial_transfer_bp_option, non_graphing_data_vectors, non_graphing_data_matrices)
                overall_y = self.graph.unflatten_initial_matrix(overall_y, [length["data"].size for length in self.graph_data.values()])
                overall_y = self.save_data(overall_y, overall_t, save_data=False)
                simulation_output.append(overall_y)
                time_output.append(overall_t)
            return self.create_initial_value_analysis_figures(simulation_output, time_output, param_name, param_1_values)
        
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
        def phase_portrait(n_clicks, param_1_name, param_2_name, param_range_1, param_steps_1, param_range_2, param_steps_2, use_serial_transfer, serial_transfer_value, serial_transfer_bp_option, serial_transfer_frequency, graphing_data, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
            #TODO: give option for auto setting arrow x and y value, give option to scale the arrow values
            #TODO: fix the issue with the arrows not pointing in the correct direction
            _, initial_condition, non_graphing_data_vectors, non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.graph.update_environment_data(environment_data[0])

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
                    initial_condition[items_of_name_1.index(param_1_name)] = X[i, j]
                    initial_condition[items_of_name_1.index(param_2_name)] = Y[i, j]
                    updated_data = self.graph.odesystem(0, initial_condition, *[self.graph, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices])
                    value1 = updated_data[items_of_name_1.index(param_1_name)]
                    value2 = updated_data[items_of_name_1.index(param_2_name)]
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
            _, initial_condition, non_graphing_data_vectors, non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.graph.update_environment_data(environment_data[0])
            updated_data = self.graph.solve_system(self.graph.odesystem, initial_condition, self.graph, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices)
            solved_y = updated_data.y
            self.copy_of_simulation_output = updated_data
            unflattened_data = self.graph.unflatten_initial_matrix(solved_y, [length["data"].size for length in self.graph_data.values()])
            unflattened_data = self.save_data(unflattened_data, updated_data.t)
            value1 = unflattened_data[items_of_name_1.index(param_1_name)][0]
            value2 = unflattened_data[items_of_name_2.index(param_2_name)][0]
            fig.add_trace(
                go.Scatter(x=value1, y=value2, mode="lines", name=f"{param_1_name} vs {param_2_name}", hovertemplate=f"{param_1_name}: %{{x}}<br>{param_2_name}: %{{y}}<br>time: %{{meta}}<extra></extra>",
                meta=updated_data.t)
            )
            return fig
        
        @callback(
            Input('save_settings', 'n_clicks'),
            Input({'type': 'settings', 'index': ALL}, 'value'),
        )
        def save_settings(n_clicks, settings):
            print(settings)

        self.app.run_server(debug=True)