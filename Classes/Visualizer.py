import numpy as np
from Classes.Analysis import Analysis
import plotly.express as px
import pandas as pd
from dash import Dash, Input, Output, callback, ALL, State
import plotly.graph_objs as go
from collections import OrderedDict
from Classes.VisualizerHTML import html_code, parse_contents
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import warnings
from copy import deepcopy
from collections import defaultdict
warnings.filterwarnings("ignore", message="The following arguments have no effect for a chosen solver: `min_step`.")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
np.random.seed(0)  # Set the random seed for reproducibility
from Classes.Math import optical_density, log_func, lin_func, serial_transfer_calculation, sum_up_columns, split_comma_minus, unifrom_color_gradient_maker

class Visualizer():
    """Class used to visualize the simulation results of the graph object. It uses the Dash library to create a web application that displays the simulation results in a user-friendly way, and allows interactivity with the data, and plotting of the data
    """
    def __init__(self, analysis: Analysis):
        """Pass an instance of an Analysis object wit

        Args:
            analysis (Analysis): The analysis object that is used to create the web application. The analysis object is used to run the simulation and report back the results. 
        """

        self.app = Dash() # the app object that is used to create the web application
        self.analysis: Analysis = analysis # 
        self.graph_data = OrderedDict()
        self.non_graph_data_vector = OrderedDict()
        self.non_graph_data_matrix = OrderedDict()
        self.other_parameters_to_pass = []
        self.copy_of_simulation_output = None
        self.copy_of_parameter_analysis_output = None
        self.settings = self.initialize_settings()
        self.initial_value_plot = OrderedDict()
        self.ending_values_serial_transfer = OrderedDict()

    def initialize_settings(self):
        """Initializes the settings data for the graph object. The settings data stores stuff like simulation length, time step, and other parameters that are used in the simulation. The settings data is stored in the graph object, and is used to run the simulation.

        Returns:
            _type_: _description_
        """
        data = self.analysis.graph.nodes["S"]["data"]
        data = parse_contents(data)
        self.analysis.settings = data
        return data
    
    def add_graph_data(self, name:str, initial_values:list, column_names:list, row_names:list=None, add_rows:bool=False):
        """Add the initial values of the graph data to the graph_data dictionary. The graph_data dictionary stores the initial values of the graph data, and is used to run the simulation. The graph_data dictionary is used to create the figures for the simulation results. 

        Args:
            name (str): The name of the grpah data. For example, it can be 'Resources', 'Uninfected Bacteria', 'Infected Bacteria', 'Phages', etc.
            initial_values (list): The initial values of the graph data. The initial values are the values that are used to run the simulation. The initial values are stored in a list, and are used to create the figures for the simulation results. It can also be a list of lists, in case you want to model intermediary steps. 
            column_names (list): Note that column_names are swapped with row_names. When plotted on the dashboard, the rows become columns, and the columns become rows. The column names are the names of the columns in the graph data, for example [B0, B1, ..., Bn] etc. 
            row_names (list, optional): Swapped with column_names, used to identify rows of multiple data. Defaults to None.
            add_rows (bool, optional): If you want to add up the rows of initial_values. Defaults to False.
        """
        self.graph_data[name] = {"data": initial_values, "column_names": column_names, "row_names": row_names, "add_rows": add_rows}

    def add_non_graph_data_vector(self, name:str, data:np.array, column_names:list):
        """Adds non-graph data (non-graph referring to data that wont be grpahed, but holds the parameter data in a vector format) to the non_graph_data_vector dictionary. The non_graph_data_vector dictionary stores the non-graph data, and is used to run the simulation. The non_graph_data_vector dictionary is used to create the figures for the simulation results.

        Args:
            name (str): Name that you want the parameter to have, for example 'tau_vector, or 'k_vector
            data (np.array): 1D np array of the data representing the parameter variable values. The data is the data that is used to run the simulation. 
            column_names (list): The column names are the names of the columns in the non-graph data, for example [B0, B1, ..., Bn] etc. The column names are used to identify the columns in the non-graph data.
        """
        self.non_graph_data_vector[name] = {"data": data, "column_names": column_names}

    def add_non_graph_data_matrix(self, name:str, data:np.array, row_names:list, column_names:list):
        """Adds non-graph data (non-graph referring to data that wont be grpahed, but holds the parameter data in a matrix format) to the non_graph_data_matrix dictionary. The non_graph_data_matrixdictionary stores the non-graph data, and is used to run the simulation. The non_graph_data_matrix dictionary is used to create the figures for the simulation results.

        Args:
            name (str): Name that you want the parameter to have, for example 'e_matrix, or 'b_matrix
            data (np.array): 2D np array of the data representing the parameter variable values. The data is the data that is used to run the simulation.
            row_names (list): The row names are the names of the rows in the non-graph data, for example [B0, B1, ..., Bn] etc. The row names are used to identify the rows in the non-graph data.
            column_names (list): The column names are the names of the columns in the non-graph data, for example [B0, B1, ..., Bn] etc. The column names are used to identify the columns in the non-graph data.
        """
        self.non_graph_data_matrix[name] = {"data": data, "row_names": row_names, "column_names": column_names}

    def add_other_parameters(self, *args):
        """Adds other parameters to the other_parameters_to_pass list. The other_parameters_to_pass list stores the other parameters that are used to run the simulation. The other_parameters_to_pass list is used to create the figures for the simulation results. 
        Pass data that you might want to explicitly use in the ODE model. 
        """
        self.other_parameters_to_pass += args
    
    def create_main_figures(self, unflattened_data, overall_t):
        """Create the main figures for the simple simulation. Shows the evolution of time and population counts. Shows the absolute and relative population levels as a bar plot, and the same for bacteria all summed up. Then shows the ending values of the inoculation and serial transfer simulation as a goruped stacked bar plot.

        Args:
            unflattened_data (list): List of numpy arrays holding the population data for each graph data.
            overall_t (np.array): Array of time values for the simulation.

        Returns:
            list: Returns a list of figures, which are the main figures for the simulation. One for the simple populaiotn evolution, one for the absolute and relative population levels, one for the absolute and relative population levels with bacteria summed up, and one for the ending values of the inoculation and serial transfer simulation.
        """
        list_of_figs = []
        num_graphs = len(self.graph_data.keys()) + 1
        cols = 3
        rows = (num_graphs // cols) + 1  # Calculate the number of rows needed
        fig = make_subplots(rows=rows, cols=3, subplot_titles=[name for name in self.graph_data.keys()] + ['Bacteria Sum (Optical Density)'], row_heights=[5000/rows]*rows)
        for i, dictionary in enumerate(self.graph_data.items()):
            name, dic = dictionary
            for j in range(len(unflattened_data[i])):
                row = (i // cols) + 1
                col = (i % cols) + 1
                fig.add_trace(go.Scatter(x=overall_t, y=unflattened_data[i][j], mode="lines", name=f"{dic['column_names'][j]}"), row=row, col=col)
                fig.update_layout(xaxis=dict(title="Time"), yaxis=dict(title="Value", type="log"))
        fig.update_layout(title=f"Graph of Each Population", xaxis=dict(title="Time"), yaxis=dict(title="Value"))

        data_bacteria = optical_density(deepcopy(unflattened_data), list(self.graph_data.keys()))
        col_index = (cols*3 + col)%3 + 1
        fig.add_trace(go.Scatter(x=overall_t, y=data_bacteria, mode="lines", name="Bacteria Sum (Optical Density)"), row=rows, col=col_index)
        fig.update_layout(hovermode="x unified")
        list_of_figs.append(fig)

        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Absolute Population Levels (Bacteria Not Summed Up)", "Relative Population Levels (Bacteria Summed Up)"), row_heights=[1000])
        np.random.seed(1)  # Set the random seed for reproducibility
        for i, (name, dic) in enumerate(self.graph_data.items()):
            for j in range(len(unflattened_data[i])):
                color = f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.8)"  # Generate a random color
                fig.add_trace(go.Scatter(x=overall_t, y=unflattened_data[i][j], mode="lines", name=f"{dic['column_names'][j]} (relative)", stackgroup="one", groupnorm='percent', marker=dict(color=color)), row=1, col=2)
        np.random.seed(1)
        for i, (name, dic) in enumerate(self.graph_data.items()):
            for j in range(len(unflattened_data[i])):
                color = f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.8)"  # Generate a random color
                fig.add_trace(go.Scatter(x=overall_t, y=unflattened_data[i][j], mode="lines", name=f"{dic['column_names'][j]} (absolute)", stackgroup="one", marker=dict(color=color)), row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="linear", ticksuffix='%', row=1, col=2)
        fig.update_layout(hovermode="x unified")
        list_of_figs.append(fig)

        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Absolute Population Levels (Bacteria Sm)", "Relative Population Levels (Bacteria Sum)"), row_heights=[1000])
        np.random.seed(1)
        for i, (name, dic) in enumerate(self.graph_data.items()):
            for j in range(len(unflattened_data[i])):
                if name.lower() in ["bacteria", "b", "u", "i", "infect", "uninf", "inf", "uninfect", "uninfected bacteria", "infected bacteria", "bacteria uninfected", "bacteria infected", "bacteria uninf", "bacteria infect"]:
                    continue
                color = f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.8)"  # Generate a random color
                fig.add_trace(go.Scatter(x=overall_t, y=unflattened_data[i][j], mode="lines", name=f"{dic['column_names'][j]} (absolute)", stackgroup="one", marker=dict(color=color)), row=1, col=1)
        color = f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.8)"  # Generate a random color
        fig.add_trace(go.Scatter(x=overall_t, y=data_bacteria, mode="lines", name="Bacteria Sum (absolute)", stackgroup="one", marker=dict(color=color)), row=1, col=1)
        np.random.seed(1)
        for i, (name, dic) in enumerate(self.graph_data.items()):
            for j in range(len(unflattened_data[i])):
                if name.lower() in ["bacteria", "b", "u", "i", "infect", "uninf", "inf", "uninfect", "uninfected bacteria", "infected bacteria", "bacteria uninfected", "bacteria infected", "bacteria uninf", "bacteria infect"]:
                    continue
                color = f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.8)"  # Generate a random color
                fig.add_trace(go.Scatter(x=overall_t, y=unflattened_data[i][j], mode="lines", name=f"{dic['column_names'][j]} (relative)", stackgroup="one", groupnorm='percent', marker=dict(color=color)), row=1, col=2)
        color = f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.8)"  # Generate a random color
        fig.add_trace(go.Scatter(x=overall_t, y=data_bacteria, mode="lines", name="Total bacteria, (optical density, relative)", stackgroup="one",  marker=dict(color=color), groupnorm='percent'), row=1, col=2)
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="linear", ticksuffix='%', row=1, col=2)
        fig.update_layout(hovermode="x unified")
        list_of_figs.append(fig)

        # Step 1: Flatten and assign numeric x positions
        records = []
        x_pos = 0
        x_labels = []
        group_gap = 0.01  # wider spacing between groups
        intra_group_gap = 0.01  # small spacing between phages and resources

        for group_label, group_data in self.ending_values_serial_transfer.items():
            group_names = group_data["group_names"]
            column_names = group_data["column_names"]
            column_data = group_data["column_data"]

            for i, (group_name, col_names, col_vals) in enumerate(zip(group_names, column_names, column_data)):
                xpos = x_pos + i * intra_group_gap
                label = f"{group_label}<br>{group_name}"
                x_labels.append((xpos, label))

                for name, val in zip(col_names, col_vals):
                    records.append({
                        "x": xpos,
                        "x_label": label,
                        "stack_label": name,
                        "value": val
                    })
            x_pos += len(group_names) * intra_group_gap + group_gap

        # Step 2: Build plotly bar traces from individual stack segments
        grouped = defaultdict(list)
        for r in records:
            grouped[r["stack_label"]].append(r)

        traces = []
        x_tick_vals = []
        x_tick_texts = []

        for stack_label, entries in grouped.items():
            traces.append(go.Bar(
                x=[e["x"] for e in entries],
                y=[e["value"] for e in entries],
                name=stack_label
            ))

        # Only unique x-labels
        for xpos, label in dict(sorted(x_labels)).items():
            x_tick_vals.append(xpos)
            x_tick_texts.append(label)

        # Step 3: Plot setup
        fig = go.Figure(data=traces)
        fig.update_layout(
            barmode="stack",
            title="Grouped + Stacked Bar Plot with Spacing",
            xaxis=dict(
                tickmode="array",
                tickvals=x_tick_vals,
                ticktext=x_tick_texts,
                tickangle=90
            ),
            yaxis_title="Value", 
            yaxis=dict(type="log"),
        )
        list_of_figs.append(fig)
        return list_of_figs

    def create_initial_value_analysis_figures(self, simulation_output, time_output, param_name, param_values, graph_axis_scale, run_name) -> list:
        """Creates the initial value analysis figures. Creates a new figure for every self.graph_data, plus one for bacteria sum. Each figure creates 3 subplots, the first one shows the classic population evolution through time. THe second one shows the starting value of the selected parameter vs the time of max value reached of the parameter. The third one shows the slope and intercept of the fitted line, and the R^2 value. The figures are created using plotly, and are returned as a list of figures. Option to have a linear or log x axis.

        Args:
            simulation_output (_type_): _description_
            time_output (_type_): _description_
            param_name (_type_): _description_
            param_values (_type_): _description_
            graph_axis_scale (_type_): _description_
            run_name (_type_): _description_

        Returns:
            list: List of figures, list length is equal to length of graph data + 1 for bacteria sum.
        """
        list_of_figs = []
        for i, name in enumerate(list(self.graph_data.keys()) + ["Bacteria Sum"]):
            fig = make_subplots(rows=1, cols=3, subplot_titles=(f"IVA for {name}", f"SV vs Time of Max Value for {name}", "Slope and Intercept Comparison"))
            list_max_x = []
            list_max_y = []
            for j in range(len(simulation_output)):
                color = unifrom_color_gradient_maker(j, len(simulation_output))
                fig.add_trace(go.Scatter(x=time_output[j], y=simulation_output[j][i][0], mode="lines", name=f"{param_name} {param_values[j]}", marker=dict(color=color)), row=1, col=1)
                max_x = time_output[j][np.argmax(simulation_output[j][i][0])]
                list_max_x.append(max_x)
                max_y = np.max(simulation_output[j][i][0])
                list_max_y.append(max_y)

            if graph_axis_scale == "linear-linear (linear)": # linear
                popt, _ = curve_fit(lin_func, param_values, list_max_x)
                predictions = [lin_func(x, *popt) for x in param_values]
                parameter_string = f"Equation: y=a*x+c<br> a: {popt[0]:.8f}<br> c: {popt[1]:.8f}<br>"
            elif graph_axis_scale == "log-linear (log)":  #log
                popt, _ = curve_fit(lin_func, np.log(param_values), list_max_x)
                predictions = [lin_func(x, *popt) for x in np.log(param_values)]
                parameter_string = f"Equation: y=a*log(x)+c<br> a: {popt[0]:.8f}<br> c: {popt[1]:.8f}<br>"
                fig.update_xaxes(type="log", row=1, col=2)

            corr_matrix = np.corrcoef(list_max_x, predictions)
            corr = corr_matrix[0,1]
            r_squared = corr**2
            fig.add_trace(
                go.Scatter(
                    x=param_values, 
                    y=predictions, 
                    mode="lines", 
                    name="Fitted Curve",
                    hovertemplate=f"Parameter Value: %{{x}}<br>Fitted Value: %{{y:.8f}}<br>" + parameter_string + f"R²: {r_squared:.8f}<extra></extra>"
                ), 
                row=1, col=2
            )
            fig.add_trace(go.Scatter(x=param_values, y=list_max_x, mode="markers", name="Data"), row=1, col=2)
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=1)
            fig.update_xaxes(title_text=f"Starting Value of {param_name}", row=1, col=2)
            fig.update_yaxes(title_text="Time max value reached", row=1, col=2)
            if name not in self.initial_value_plot:
                self.initial_value_plot[name] = {
                    'data': [[popt[0], popt[1], r_squared]],
                    'iterations': ["Run " + str(1) if run_name == "" else run_name],
                }
            else:
                self.initial_value_plot[name]['data'] += [[popt[0], popt[1], r_squared]]
                self.initial_value_plot[name]['iterations'] += ["Run " + str(len(self.initial_value_plot[name]['iterations']) + 1) if run_name == "" else run_name]

            for j in range(len(self.initial_value_plot[name]['data'])):
                for k, value in enumerate(self.initial_value_plot[name]['data'][j]):
                    self.initial_value_plot[name]['data'][j][k] = round(value, 9)
                fig.add_trace(
                    go.Bar(
                        x=["a", "c", "R^2"], 
                        y=self.initial_value_plot[name]['data'][j], 
                        name=self.initial_value_plot[name]['iterations'][j], 
                        text=self.initial_value_plot[name]['data'][j],
                        textposition=["outside" if value >= 0 else "outside" for value in self.initial_value_plot[name]['data'][j]],
                        textangle=-90  # Set text orientation to vertical
                    ), 
                    row=1, col=3
                )
            list_of_figs.append(fig)
        return list_of_figs
    
    def create_numpy_lists(self, graphing_data, graphing_data_vectors, graphing_data_matrices):
        graphing_data = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data]
        flattened = self.analysis.flatten_lists_and_matrices(*graphing_data)
        non_graphing_data_vectors = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy()[0] for data_values in graphing_data_vectors]
        non_graphing_data_matrices = [pd.DataFrame.from_dict(data_values).astype(float).to_numpy() for data_values in graphing_data_matrices]
        return graphing_data, flattened, non_graphing_data_vectors, non_graphing_data_matrices
        
    def save_data(self, array, time, save_data=True):
        unflattened_data = []
        for dic, data_item in zip(self.graph_data.items(), array):
            key, value = dic
            if save_data:
                self.graph_data[key]["y_data"] = data_item
                self.graph_data[key]["t_data"] = time
            unflattened_data.append(sum_up_columns(data_item, value["add_rows"]))
        return unflattened_data
    
    def create_heatmap_figures(self, matrix_data, x_axis_data=None, y_axis_data=None, x_labels=None, y_labels=None):
        list_of_figs = []
        for i, name in zip(range(matrix_data.shape[2]), list(self.graph_data.keys()) + ["Bacteria Sum"]):
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
        
    def run_serial_transfer_iterations(self, overall_y, overall_t, serial_transfer_frequency, flattened, serial_transfer_value, serial_transfer_bp_option, non_graphing_data_vectors, non_graphing_data_matrices, save_bar_plot=False):
        """Runs a serial transfer simulation for the given number of iterations. The serial transfer simulation is a simulation that runs for a given number of iterations, and saves the data to the graph_data dictionary. The serial transfer simulation is used to create the figures for the simulation results.

        Args:
            overall_y (np array): np array of the population levels, straight out of the ODE solver, no other processing done to it.
            overall_t (np array): np array of the time values, straight out of the ODE solver, no other processing done to it.
            serial_transfer_frequency (int): The number of iterations to run the serial transfer simulation for. 
            flattened (_type_): _description_
            serial_transfer_value (_type_): _description_
            serial_transfer_bp_option (_type_): _description_
            non_graphing_data_vectors (_type_): _description_
            non_graphing_data_matrices (_type_): _description_
            save_bar_plot (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        final_values = overall_y[:, -1]
        final_time = overall_t[-1]
        for _ in range(int(serial_transfer_frequency)):
            flattened = serial_transfer_calculation(self.graph_data, final_values, serial_transfer_value, serial_transfer_bp_option, flattened)
            solved_system = self.analysis.solve_system(self.analysis.odesystem, flattened, self.analysis, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices, t_start=float(final_time), t_end=float(final_time) + float(self.settings['simulation_length']))
            overall_y = np.concatenate((overall_y, solved_system.y), axis=1)
            overall_t = np.concatenate((overall_t, solved_system.t))
            if save_bar_plot:
                overall_y_2 = self.analysis.unflatten_initial_matrix(solved_system.y, [length["data"].size for length in self.graph_data.values()])
                overall_y_2 = self.save_data(overall_y_2, solved_system.t, save_data=False)
                column_data = []
                for i, name in enumerate(list(self.graph_data.keys())):
                    temp_list = []
                    for j in range(len(overall_y_2[i])):
                        temp_list.append(overall_y_2[i][j][-1])
                    column_data.append(temp_list)
                data_bacteria = optical_density(deepcopy(overall_y_2), list(self.graph_data.keys()))
                column_data.append([data_bacteria[-1]])
                group_names = [name for name in self.graph_data.keys()] + ["Bacteria Sum"]
                column_names = [self.graph_data[name]["column_names"] for name in self.graph_data.keys()] + [["Bacteria Sum"]]
                self.ending_values_serial_transfer["ST " + str(len(self.ending_values_serial_transfer))] = {'group_names': group_names, 'column_names': column_names, 'column_data': column_data}
            final_values = solved_system.y[:, -1]
            final_time = solved_system.t[-1]
        return overall_y, overall_t
    
    def run(self):
        """Runs the Dash application. The Dash application is a web application that displays the simulation results in a user-friendly way, and allows interactivity with the data, and plotting of the data. The Dash application is run on the local host, and can be accessed from the web browser. Needs to ahve the data loaded properly in using the add_graph_data, add_non_graph_data_vector, add_non_graph_data_matrix, and add_other_parameters methods. 
        """
        self.app.layout = html_code(self.graph_data, self.non_graph_data_vector, self.non_graph_data_matrix, self.analysis, self.settings)

        @callback(
            Output({'type': 'plot_basic_graph_data', 'index': "plot_basic_graph_data"}, 'figure', allow_duplicate=True),
            Output({'type': 'plot_basic_graph_data', 'index': "plot_basic_graph_data_total_sum"}, 'figure', allow_duplicate=True),
            Output({'type': 'plot_basic_graph_data', 'index': "plot_basic_graph_data_bacteria_sum_graph"}, 'figure', allow_duplicate=True),
            Output({'type': 'plot_basic_graph_data', 'index': "plot_basic_graph_serial_transfer_end_values"}, 'figure', allow_duplicate=True),
            Input('run_basic_model', 'n_clicks'),
            State({'type': 'edit_graphing_data', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_vectors', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_matrices', 'index': ALL}, 'data'),
            State('environment_data', 'data'),
            prevent_initial_call=True
        )
        def plot_main_plots(n_clicks, graphing_data, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
            """_summary_

            Args:
                n_clicks (int): The number of times the save button has been clicked. Used to trigger the callback. 
                graphing_data (_type_): _description_
                non_graphing_data_vectors (_type_): _description_
                non_graphing_data_matrices (_type_): _description_
                environment_data (_type_): _description_

            Returns:
                Returns:
                list: List of figures, list length is equal to length of graph data + 1 for bacteria sum.
            """
            # turn the data in the dashboard into numpy arrays, and save/update the environemnt data to the graph object
            _, flattened, non_graphing_data_vectors, non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.analysis.update_environment_data(environment_data[0])

            # solves sytem of ODEs, saves y and t data results
            solved_system = self.analysis.solve_system(self.analysis.odesystem, flattened, self.analysis, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices)
            self.copy_of_simulation_output = solved_system

            # unflattens the data, saves the data to the graph_data dictionary, and sums up the columns if necessary
            overall_y = self.analysis.unflatten_initial_matrix(solved_system.y, [length["data"].size for length in self.graph_data.values()])
            overall_y = self.save_data(overall_y, solved_system.t)
            column_data = []
            for i, name in enumerate(list(self.graph_data.keys())):
                temp_list = []
                for j in range(len(overall_y[i])):
                    temp_list.append(overall_y[i][j][-1])
                column_data.append(temp_list)
            data_bacteria = optical_density(deepcopy(overall_y), list(self.graph_data.keys()))
            column_data.append([data_bacteria[-1]])
            self.ending_values_serial_transfer = {}
            group_names = [name for name in self.graph_data.keys()] + ["Bacteria Sum"]
            column_names = [self.graph_data[name]["column_names"] for name in self.graph_data.keys()] + [["Bacteria Sum"]]
            self.ending_values_serial_transfer["Initial Inoculation"] = {'group_names': group_names, 'column_names': column_names, 'column_data': column_data}
            list_of_figs = self.create_main_figures(overall_y, solved_system.t)
            return list_of_figs
        
        @callback(
            Output({'type': 'plot_basic_graph_data', 'index': 'plot_basic_graph_data'}, 'figure', allow_duplicate=True),
            Output({'type': 'plot_basic_graph_data', 'index': "plot_basic_graph_data_total_sum"}, 'figure', allow_duplicate=True),
            Output({'type': 'plot_basic_graph_data', 'index': "plot_basic_graph_data_bacteria_sum_graph"}, 'figure', allow_duplicate=True),
            Output({'type': 'plot_basic_graph_data', 'index': "plot_basic_graph_serial_transfer_end_values"}, 'figure', allow_duplicate=True),
            Input('run_serial_transfer', 'n_clicks'),
            State('serial_transfer_value', 'value'),
            State('serial_transfer_bp_option', 'value'),
            State('serial_transfer_frequency', 'value'),
            State({'type': 'edit_graphing_data', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_vectors', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_matrices', 'index': ALL}, 'data'),
            State('environment_data', 'data'),
            prevent_initial_call=True
        )
        def serial_transfer(n_clicks, serial_transfer_value, serial_transfer_bp_option, serial_transfer_frequency, graphing_data, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
            """_summary_

            Args:
                n_clicks (int): The number of times the save button has been clicked. Used to trigger the callback. 
                serial_transfer_value (float): The value of the serial transfer to divide the population by.
                serial_transfer_bp_option (list): The option to add phages and bacteria to the serial transfer.
                serial_transfer_frequency (int): Number of times to automatically run serial transfer
                graphing_data (list): A list of lists. Inside each list, a dictionary with the graph data in key:value format for the individual sub-resources, phages, etc. 
                non_graphing_data_vectors (list): A list of lists. Inside each list, a dictionary with the non-graphing-vector data in key:value format. Key name is the column of the table.
                non_graphing_data_matrices (list): A list of lists. Inside each list, a dictionary with the non-graphing-matrix data in key:value format. Key name is the column of the table. 
                environment_data (list): A list with a single dictionary holding the environment data in key:value format

            Returns:
                list: List of figures, list length is equal to length of graph data + 1 for bacteria sum.
            """
            print(graphing_data)
            print(non_graphing_data_vectors)
            print(non_graphing_data_matrices) 
            print(environment_data)
            # turn the data in the dashboard into numpy arrays, and save/update the environemnt data to the graph object
            _, initial_condition, new_non_graphing_data_vectors, new_non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.analysis.update_environment_data(environment_data[0])

            #use previously saved data to start the simulation from the last time point
            try:
                overall_t = self.copy_of_simulation_output.t
            except:
                list_figures = [go.Figure() for _ in self.graph_data.keys()]
                list_figures[0].update_layout(title="No graph to serial transfer, please run the simulation first usign the 'Rerun model' button below")
                return list_figures

            overall_y = self.copy_of_simulation_output.y
            
            # for the required number of runs of serial transfer
            overall_y, overall_t = self.run_serial_transfer_iterations(overall_y, overall_t, serial_transfer_frequency, initial_condition, serial_transfer_value, serial_transfer_bp_option, new_non_graphing_data_vectors, new_non_graphing_data_matrices, save_bar_plot=True)
            
            # save the values to self.copy_of_simulation_output.y/t respectively, in case for future serial transfers
            self.copy_of_simulation_output.t = overall_t
            self.copy_of_simulation_output.y = overall_y

            # unflatten the data, save the data to the graph_data dictionary, and sum up the columns if necessary, then create the figures
            overall_y = self.analysis.unflatten_initial_matrix(overall_y, [length["data"].size for length in self.graph_data.values()])
            overall_y = self.save_data(overall_y, overall_t, save_data=False)
            list_of_figs = self.create_main_figures(overall_y, overall_t)
            return list_of_figs
        
        @callback(
            [Output({'type': 'plot_parameter_analysis', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()], 
            Output({'type': 'plot_parameter_analysis', 'index': "plot_parameter_analysis_bacteria_sum"}, 'figure', allow_duplicate=True),
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
            State('serial_transfer_bp_option', 'value'),
            State('serial_transfer_frequency', 'value'),
            State({'type': 'edit_graphing_data', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_vectors', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_matrices', 'index': ALL}, 'data'),
            State('environment_data', 'data'),
            prevent_initial_call=True
        )
        def parameter_analysis(n_clicks, param_name_1, param_name_2, use_opt_1_or_opt_2, param_1_input, param_2_input, param_range_1, param_range_2, param_steps_1, param_steps_2, use_serial_transfer, serial_transfer_value, serial_transfer_bp_option, serial_transfer_frequency, graphing_data, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
            """_summary_

            Args:
                n_clicks (int): The number of times the save button has been clicked. Used to trigger the callback. 
                param_name_1 (_type_): _description_
                param_name_2 (_type_): _description_
                use_opt_1_or_opt_2 (_type_): _description_
                param_1_input (_type_): _description_
                param_2_input (_type_): _description_
                param_range_1 (_type_): _description_
                param_range_2 (_type_): _description_
                param_steps_1 (_type_): _description_
                param_steps_2 (_type_): _description_
                use_serial_transfer (list): If want to use serial transfer option, list ['option1'] is returned for yes, else it is [] for no. 
                serial_transfer_value (float): The value of the serial transfer to divide the population by.
                serial_transfer_bp_option (list): The option to add phages and bacteria to the serial transfer.
                serial_transfer_frequency (int): Number of times to automatically run serial transfer
                graphing_data (list): A list of lists. Inside each list, a dictionary with the graph data in key:value format for the individual sub-resources, phages, etc. 
                non_graphing_data_vectors (list): A list of lists. Inside each list, a dictionary with the non-graphing-vector data in key:value format. Key name is the column of the table.
                non_graphing_data_matrices (list): A list of lists. Inside each list, a dictionary with the non-graphing-matrix data in key:value format. Key name is the column of the table. 
                environment_data (list): A list with a single dictionary holding the environment data in key:value format. 

            Returns:
                _type_: _description_
            """
            # turn the data in the dashboard into numpy arrays, and save/update the environemnt data to the graph object
            _, initial_condition, non_graphing_data_vectors, non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.analysis.update_environment_data(environment_data[0])

            #if option 1 is selected, then the values used to test the simulation are split by commas, and are put into a list as a float. Otherwise the range is split by a dash and linspace is used to create the values, and put into a list as a float
            try:
                param_values_1 = split_comma_minus(param_1_input, param_range_1, param_steps_1, use_opt_1_or_opt_2)
                param_values_2 = split_comma_minus(param_2_input, param_range_2, param_steps_2, use_opt_1_or_opt_2)
            except:
                return *[go.Figure() for _ in self.graph_data.keys()], go.Figure(), 0, {}, 0

            # create a matrix to store the values of the final time point for each parameter value tested
            matrix_output = np.zeros((len(param_values_1), len(param_values_2), len(self.graph_data)+1))
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
                    solved_system = self.analysis.solve_system(self.analysis.odesystem, initial_condition, self.analysis, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices)
                    overall_y = solved_system.y
                    overall_t = solved_system.t
                    
                    # if serial transfer is selected, then the system is run for the number of iterations specified, and the final values are saved
                    if use_serial_transfer:
                        overall_y, overall_t = self.run_serial_transfer_iterations(overall_y, overall_t, serial_transfer_frequency, initial_condition, serial_transfer_value, serial_transfer_bp_option, non_graphing_data_vectors, non_graphing_data_matrices)
                    overall_y = self.analysis.unflatten_initial_matrix(overall_y, [length["data"].size for length in self.graph_data.values()])
                    overall_y = self.save_data(overall_y, overall_t, save_data=False)
                    overall_y.append([optical_density(deepcopy(overall_y), list(self.graph_data.keys()))])
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
            Output({'type': 'plot_parameter_analysis', 'index': "plot_parameter_analysis_bacteria_sum"}, 'figure', allow_duplicate=True),
            # Input('parameter_analysis_slider', 'drag_value'),
            Input('parameter_analysis_slider', 'value'),
            Input('parameter_analysis_extrapolate', 'value'),
            prevent_initial_call=True
        )
        def parameter_analysis_slider_update(slider_value, extrapolate):
            """_summary_

            Args:
                slider_value (_type_): _description_
                extrapolate (_type_): _description_

            Returns:
                _type_: _description_
            """
            # when first launching the app and opening the Parameter Analysis tab, error is thrown by dash for self.copy_of_parameter_analysis_output being None/empty, so return empty figures to avoid/fix/alleviate this error
            # collect all the stored data from the parameter analysis, and create a heatmap of the final values for each parameter value
            try:
                param_values_1 = self.copy_of_parameter_analysis_output["x_axis_data"]
            except:
                return [go.Figure() for _ in self.graph_data.keys()] + [go.Figure()]
            param_values_2 = self.copy_of_parameter_analysis_output["y_axis_data"]
            param_name_1 = self.copy_of_parameter_analysis_output["x_labels"]
            param_name_2 = self.copy_of_parameter_analysis_output["y_labels"]
            overall_t = self.copy_of_parameter_analysis_output["overall_t"]
            overall_y = self.copy_of_parameter_analysis_output["overall_y"]

            # create a matrix to store the values of the final time point for each parameter value tested
            matrix_output = np.zeros((len(param_values_1), len(param_values_2), len(self.graph_data)+1))
            
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

        @callback(
            [Output({'type': 'plot_initial_value_analysis', 'index': name}, 'figure', allow_duplicate=True) for name in self.graph_data.keys()],
            Output({'type': 'plot_initial_value_analysis', 'index': "plot_initial_value_analysis_bacteria_sum"}, 'figure', allow_duplicate=True),
            Input('run_initial_value_analysis', 'n_clicks'),
            State('initial_value_analysis_param_name', 'value'),
            State('initial_value_analysis_option', 'value'),
            State('initial_value_analysis_input', 'value'),
            State('initial_value_analysis_range', 'value'),
            State('initial_value_analysis_steps', 'value'),
            State('initial_value_analysis_run_name', 'value'),
            State('initial_value_analysis_use_serial_transfer', 'value'),
            State('serial_transfer_value', 'value'),
            State('serial_transfer_bp_option', 'value'),
            State('serial_transfer_frequency', 'value'),
            State('initial_value_analysis_graph_scale', 'value'),
            State({'type': 'edit_graphing_data', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_vectors', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_matrices', 'index': ALL}, 'data'),
            State('environment_data', 'data'),
            prevent_initial_call=True
        )
        def initial_value_analysis(n_clicks, param_name, use_opt_1_or_opt_2, param_input, param_range, param_steps, run_name, use_serial_transfer, serial_transfer_value, serial_transfer_bp_option, serial_transfer_frequency, graph_axis_scale, graphing_data, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
            """_summary_

            Args:
                n_clicks (int): The number of times the save button has been clicked. Used to trigger the callback. 
                param_name (_type_): _description_
                use_opt_1_or_opt_2 (_type_): _description_
                param_input (_type_): _description_
                param_range (_type_): _description_
                param_steps (_type_): _description_
                run_name (_type_): _description_
                use_serial_transfer (list): If want to use serial transfer option, list ['option1'] is returned for yes, else it is [] for no. 
                serial_transfer_value (float): The value of the serial transfer to divide the population by.
                serial_transfer_bp_option (list): The option to add phages and bacteria to the serial transfer.
                serial_transfer_frequency (int): Number of times to automatically run serial transfer
                graphing_data (list): A list of lists. Inside each list, a dictionary with the graph data in key:value format for the individual sub-resources, phages, etc. 
                non_graphing_data_vectors (list): A list of lists. Inside each list, a dictionary with the non-graphing-vector data in key:value format. Key name is the column of the table.
                non_graphing_data_matrices (list): A list of lists. Inside each list, a dictionary with the non-graphing-matrix data in key:value format. Key name is the column of the table. 
                environment_data (list): A list with a single dictionary holding the environment data in key:value format. 

            Returns:
                _type_: _description_
            """
            # turn the data in the dashboard into numpy arrays, and save/update the environemnt data to the graph object
            _, initial_condition, non_graphing_data_vectors, non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.analysis.update_environment_data(environment_data[0])

            # if option 1 is selected, then the values used to test the simulation are split by commas, and are put into a list as a float. Otherwise the range is split by a dash and linspace is used to create the values, and put into a list as a float
            param_1_values = split_comma_minus(param_input, param_range, param_steps, use_opt_1_or_opt_2)
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
                updated_data = self.analysis.solve_system(self.analysis.odesystem, initial_condition, self.analysis, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices)
                overall_y = updated_data.y
                overall_t = updated_data.t
                if use_serial_transfer:
                    overall_y, overall_t = self.run_serial_transfer_iterations(overall_y, overall_t, serial_transfer_frequency, initial_condition, serial_transfer_value, serial_transfer_bp_option, non_graphing_data_vectors, non_graphing_data_matrices)
                overall_y = self.analysis.unflatten_initial_matrix(overall_y, [length["data"].size for length in self.graph_data.values()])
                overall_y = self.save_data(overall_y, overall_t, save_data=False)
                overall_y.append([optical_density(deepcopy(overall_y), list(self.graph_data.keys()))])
                simulation_output.append(overall_y)
                time_output.append(overall_t)
            return self.create_initial_value_analysis_figures(simulation_output, time_output, param_name, param_1_values, graph_axis_scale, run_name)
        
        @callback(
            Output('plot_phase_portrait', 'figure', allow_duplicate=True),
            Input('run_phase_portrait', 'n_clicks'),
            State('phase_portrait_param_name_1', 'value'),
            State('phase_portrait_param_name_2', 'value'),
            State('phase_portrait_range_1', 'value'),
            State('phase_portrait_steps_1', 'value'),
            State('phase_portrait_range_2', 'value'),
            State('phase_portrait_steps_2', 'value'),
            State('phase_portrait_starting_x', 'value'),
            State('phase_portrait_starting_y', 'value'),
            State('phase_portrait_auto_calculate_range', 'value'),
            State('phase_portrait_use_serial_transfer', 'value'),
            State('serial_transfer_value', 'value'),
            State('serial_transfer_bp_option', 'value'),
            State('serial_transfer_frequency', 'value'),
            State({'type': 'edit_graphing_data', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_vectors', 'index': ALL}, 'data'),
            State({'type': 'edit_non_graphing_data_matrices', 'index': ALL}, 'data'),
            State('environment_data', 'data'),
            prevent_initial_call=True
        )
        def phase_portrait(n_clicks, param_name_1, param_name_2, param_range_1, param_steps_1, param_range_2, param_steps_2, starting_x, starting_y, auto_calculate_range, use_serial_transfer, serial_transfer_value, serial_transfer_bp_option, serial_transfer_frequency, graphing_data, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
            """Creates a phase portrait of a simulation. Choose two parameters to plot against each other, and the simulation will be run for each combination of the two parameters. The phase portrait is a 2D plot of the two parameters, with the x-axis being the first parameter and the y-axis being the second parameter. The phase portrait is created by running the simulation for each combination of the two parameters, and plotting the results.

            Args:
                n_clicks (int): _number of clicks on the button_
                param_name_1 (str): _description_
                param_name_2 (str): _description_
                param_range_1 (str): _description_
                param_steps_1 (int): _description_
                param_range_2 (str): _description_
                param_steps_2 (int): _description_
                starting_x (str): _description_
                starting_y (str): _description_
                auto_calculate_range (list): _description_
                use_serial_transfer (list): If want to use serial transfer option, list ['option1'] is returned for yes, else it is [] for no. 
                serial_transfer_value (float): The value of the serial transfer to divide the population by.
                serial_transfer_bp_option (list): The option to add phages and bacteria to the serial transfer.
                serial_transfer_frequency (int): Number of times to automatically run serial transfer
                graphing_data (list): A list of lists. Inside each list, a dictionary with the graph data in key:value format for the individual sub-resources, phages, etc. 
                non_graphing_data_vectors (list): A list of lists. Inside each list, a dictionary with the non-graphing-vector data in key:value format. Key name is the column of the table.
                non_graphing_data_matrices (list): A list of lists. Inside each list, a dictionary with the non-graphing-matrix data in key:value format. Key name is the column of the table. 
                environment_data (list): A list with a single dictionary holding the environment data in key:value format. 

            Returns:
                _type_: _description_
            """
            _, initial_condition, non_graphing_data_vectors, non_graphing_data_matrices = self.create_numpy_lists(graphing_data, non_graphing_data_vectors, non_graphing_data_matrices)
            self.analysis.update_environment_data(environment_data[0])
            starting_x = [float(value.strip()) for value in starting_x.split(",")]
            starting_y = [float(value.strip()) for value in starting_y.split(",")]
            items_of_name_full = []
            items_of_name_short = []
            for key, value in self.graph_data.items():
                items_of_name_full += [key] * value["data"].size
                items_of_name_short.append(key)
            min_x_overall, max_x_overall = None, None
            min_y_overall, max_y_overall = None, None
            list_of_solved = []
            for i in range(len(starting_x)):
                for j in range(len(starting_y)):
                    index = items_of_name_full.index(param_name_1)
                    initial_condition[index] = starting_x[i]
                    index = items_of_name_full.index(param_name_2)
                    initial_condition[index] = starting_y[j]
                    solved_system = self.analysis.solve_system(self.analysis.odesystem, initial_condition, self.analysis, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices)
                    overall_y = solved_system.y
                    overall_t = solved_system.t
                    if use_serial_transfer:
                        overall_y, overall_t = self.run_serial_transfer_iterations(overall_y, overall_t, serial_transfer_frequency, initial_condition, serial_transfer_value, serial_transfer_bp_option, non_graphing_data_vectors, non_graphing_data_matrices)
                    self.copy_of_simulation_output = solved_system
                    unflattened_data = self.analysis.unflatten_initial_matrix(overall_y, [length["data"].size for length in self.graph_data.values()])
                    unflattened_data = self.save_data(unflattened_data, overall_t)
                    solved_x_values = unflattened_data[items_of_name_short.index(param_name_1)][0]
                    solved_y_values = unflattened_data[items_of_name_short.index(param_name_2)][0]
                    list_of_solved.append((solved_x_values, solved_y_values, overall_t, starting_x[i], starting_y[j]))
                    if (auto_calculate_range):
                        min_x, max_x = min(solved_x_values), max(solved_x_values)
                        min_y, max_y = min(solved_y_values), max(solved_y_values)
                    else:
                        min_x, max_x = [float(value.strip()) for value in param_range_1.split("-")]
                        min_y, max_y = [float(value.strip()) for value in param_range_2.split("-")]
                    if (min_x_overall is None or min_x < min_x_overall):
                        min_x_overall = min_x
                    if (max_x_overall is None or max_x > max_x_overall):
                        max_x_overall = max_x
                    if (min_y_overall is None or min_y < min_y_overall):
                        min_y_overall = min_y
                    if (max_y_overall is None or max_y > max_y_overall):
                        max_y_overall = max_y
            x_vals = np.linspace(float(min_x_overall), float(max_x_overall), int(param_steps_1))
            y_vals = np.linspace(float(min_y_overall), float(max_y_overall), int(param_steps_2))
            X, Y = np.meshgrid(x_vals, y_vals)

            DX, DY = np.zeros(X.shape), np.zeros(Y.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    initial_condition_copy = initial_condition.copy()
                    initial_condition_copy[items_of_name_full.index(param_name_1)] = X[i, j]
                    initial_condition_copy[items_of_name_full.index(param_name_2)] = Y[i, j]
                    updated_data = self.analysis.odesystem(0, initial_condition_copy, *[self.analysis, *self.other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices])
                    value1 = updated_data[items_of_name_full.index(param_name_1)]
                    value2 = updated_data[items_of_name_full.index(param_name_2)]
                    DX[i, j], DY[i, j] = value1, value2

            fig = go.Figure()
            fig.update_layout(
                title=f"Phase Portrait for {param_name_1} vs {param_name_2}",
                xaxis=dict(title=param_name_1),
                yaxis=dict(title=param_name_2),
                width=1200, 
                height=800
            )
            list_x = []
            list_y = []
            for solved in list_of_solved:
                fig.add_trace(
                    go.Scatter(
                        x=solved[0], y=solved[1], mode="lines", 
                        name=f"Starting Point: ({solved[3]}, {solved[4]})", 
                        hovertemplate=f"{param_name_1}: %{{x}}<br>{param_name_2}: %{{y}}<br>time: %{{meta}}<br>Initial Condition: ({solved[3]}, {solved[4]})<extra></extra>",      
                        meta=solved[2]
                    )
                )
                list_x.append(solved[3])
                list_y.append(solved[4])
            fig.add_trace(
                go.Scatter(
                    x=list_x, y=list_y, mode="markers", 
                    showlegend=False,
                    hovertemplate=f"{param_name_1}: %{{x}}<br>{param_name_2}: %{{y}}<extra></extra>",      
                )
            )
            return fig
        
        @callback(
            Input('save_settings', 'n_clicks'),
            Input({'type': 'settings', 'index': ALL}, 'id'),
            Input({'type': 'settings', 'index': ALL}, 'value'),
        )
        def save_settings(n_clicks, settings_name, settings_value):
            """Saves the settings from the dashboard to the analysis object, and updates the settings in the analysis object. 

            Args:
                n_clicks (int): The number of times the save button has been clicked. Used to trigger the callback. 
                settings_name (dict): Dictionary of the settings names, used to find the index of the setting in the analysis object.
                settings_value (list): Values of the settings, used to update the settings in the analysis object.
            """
            new_settings = {} # new settings dictionary to save the settings to
            for i in range(len(settings_name)): # loop through the settings names and rename, makes it easier in the enxt step
                settings_name[i] = settings_name[i]['index']
            for name, value in zip(settings_name, settings_value): # loop through the settings names and values, and save them to the new settings dictionary
                if type(value) == list: # this is used for the option checkbox, if checked in the dashbaord, the value retunred is ['option'], otherwise it is []
                    value = True if value != [] else False
                new_settings[name] = value # save the value to the new settings dictionary
            self.analysis.settings = new_settings # save new_settings to the analysis object
            self.settings = new_settings # save the new settings to the class variable

        @callback(
            [Output({'type': 'plot_initial_value_analysis', 'index': name}, 'figure') for name in self.graph_data.keys()],
            Output({'type': 'plot_initial_value_analysis', 'index': 'plot_initial_value_analysis_bacteria_sum'}, 'figure'),
            Input('clear_bar_chart', 'n_clicks'),
        )
        def clear_bar_chart(n_clicks):
            """Clears the bar chart data from the dashboard, and resets the bar chart to empty figures.

            Args:
                n_clicks (int): number of times the clear button has been clicked. Used to trigger the callback. 

            Returns:
                list: list of length self.graph_data.keys() + 1 (for bacteria sum) of empty go.Figure() objects to reset the bar chart.
            """
            self.initial_value_plot = {}
            return [go.Figure() for _ in self.graph_data.keys()] + [go.Figure()]

        # run the app
        self.app.run(debug=True)