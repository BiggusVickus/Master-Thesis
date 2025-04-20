from dash import Dash, dash_table, html, Input, Output, callback, ALL, State
from dash import dcc
from collections import OrderedDict
import pandas as pd
from Classes.Analysis import Analysis

def parse_contents(contents:str) -> dict:
    """Parses the contents of a txt file (passed as a string) and returns a dictionary with the contents.
    Accepted data:
    - True/False
    - int 1
    - list [1, 2, 3] of either ints or floats
    - float 1.0
    - None
    - string "string"
    - empty string "" returns None

    Args:
        contents (str): The contents of the txt file as a string. Represented as a string with each line containing a key-value pair separated by a colon, each line is separated by a newline character '\\n'.

    Returns:
        dict: Dictionary with the contents of the txt file. The keys are the content types and the values are the content values. Auto detects the correct type of the value, eg True/False, float, int, list, None, string.
    """
    dictionary = {}
    for line in contents.splitlines():
        if line.strip():
            content_type, content = line.split(":")
            content = content.strip()
            if content == "True":
                content = True
            elif content == "False":
                content = False
            elif "[" in content:
                content = content.replace('[', '').replace(']', '').replace(' ', '')
                content = content.split(',')
                content = [int(element) if element.isnumeric() else float(element) for element in content]
            elif "." in content:
                content = float(content)
            elif content.isnumeric():
                content = int(content)
            elif content == "None":
                content = None
            elif content == "":
                content = None
            else:
                content = content
            dictionary[content_type] = content
        dictionary[content_type] = content
    return dictionary

def html_code(graph_data, non_graph_data_vector, non_graph_data_matrix, analysis:Analysis, initial_settings):
    """Creates the HTML for the dashboard. 

    Args:
        graph_data (dict): Ordered dictionary with the graph data. The keys are the names of the data and the values are dictionaries with the data, row names, and column names. The data is a list of lists or a matrix. The row names are a list of strings and the column names are a list of strings.
        non_graph_data_vector (dict): Ordered dictionary with the non graph data. The keys are the names of the data and the values are dictionaries with the data, row names, and column names. The data is a list of lists or a matrix. The row names are a list of strings and the column names are a list of strings.
        non_graph_data_matrix (dict): Odered dictionary with the non graph data. The keys are the names of the data and the values are dictionaries with the data, row names, and column names. The data is a list of lists or a matrix. The row names are a list of strings and the column names are a list of strings.
        analysis (Analysis): Instance of the Analysis class. 
        initial_settings (dict): Dictionary with the initial settings. The keys are the names of the settings and the values are the values of the settings. The settings are solver type, min step size, max step size, cutoff value, dense output, relative tolerance, absolute tolerance, and simulation length.
    """
    graph_data_name_list = [name for name in graph_data.keys()]
    non_graph_data_name_list = [name for name in OrderedDict(list(non_graph_data_vector.items()) + list(non_graph_data_matrix.items()))]
    environment_params = [name for name in analysis.environment_data.keys()]
    both_params = graph_data_name_list + non_graph_data_name_list + environment_params

    return html.Div([
            # main title and area for main, basic plots. 1 for the basic plot, 1 for the bacteria not sum stacked lineplot, 1 for the bacteria summed up stacked lineplot, and 1 for the serial transfer end values plot
            html.H1("Line Chart"),
            dcc.Graph(id={"type": "plot_basic_graph_data", "index": "plot_basic_graph_data"}), 
            dcc.Graph(id={"type": "plot_basic_graph_data", "index": "plot_basic_graph_data_total_sum"}), 
            dcc.Graph(id={"type": "plot_basic_graph_data", "index": "plot_basic_graph_data_bacteria_sum_graph"}), 
            dcc.Graph(id={"type": "plot_basic_graph_data", "index": "plot_basic_graph_serial_transfer_end_values"}), 
            # break, formatting, and button to save and rerun the model
            html.Div(style={'margin': '60px'}),
            html.Hr(),
            html.Div(style={'margin': '60px'}),
            dcc.Checklist(
                options=[
                    {'label': 'Use log y axis (checked) or linear y axis (unchecked)', 'value': 'option1'},
                ],
                value=['option1'],
                id='main_figure_log_axis'
            ),
            html.Button('Rerun model', id='run_basic_model'),

            # tabs for the data tables, the initial conditions, parameter values for the simulation, environment parameters, and the settings tab
            dcc.Tabs([
                # initial conditions: loop through the graph data and create a DataTable for each one to show initial conditions
                dcc.Tab(label='Graphing Data (Initial Conditions)', children=[
                    *[
                        html.Div([
                            html.H2(f"DataTable for {name}"),
                            html.H3(f"Row names {dic['row_names']}") if dic["row_names"] is not None else None, # in case the row names are not None
                            dash_table.DataTable(
                                # first turn into dataframe. there is the [dic["data"]], which is a list of lists, and the column names, which is a list of strings. If the row names are None, then we just use the data as is. This isfor vector or matrix representation respectively. 
                                pd.DataFrame([dic["data"]], columns=dic["column_names"]).to_dict('records') if dic["row_names"] is None else pd.DataFrame(dic["data"], columns=dic["column_names"]).to_dict('records'),
                                id={"type": 'edit_graphing_data', 'index': name},
                                columns=[{'name': f"{col}", 'id': col} for col in dic["column_names"]],
                                editable=True
                            ),
                        ]) for name, dic in graph_data.items()
                    ]
                ]),
                
                # non graphing data: loop through the non graphing data and create a DataTable for each one to show parameter values
                dcc.Tab(label='Non Graphing Data (Parameter Values): Vectors', children=[
                    *[
                        html.Div([
                            html.H2(f"DataTable for {name}"),
                            dash_table.DataTable(
                                # no need for dic["row_names"] here, since it is a vector
                                pd.DataFrame([dic["data"]], columns=dic["column_names"]).to_dict('records'),
                                id={"type": 'edit_non_graphing_data_vectors', 'index': name},
                                columns=[{'name': f"{col}", 'id': col} for col in dic["column_names"]],
                                editable=True
                            ),
                        ]) for name, dic in non_graph_data_vector.items()
                    ]
                ]),
                
                # non graphing data: loop through the non graphing data and create a DataTable for each one to show parameter values
                dcc.Tab(label='Non Graphing Data (Parameter Values): Matrices', children=[
                    *[
                        html.Div([
                            html.H2(f"DataTable for {name}"),
                            html.H3(f"Row Names: {dic['row_names']}"),
                            dash_table.DataTable(
                                # no need for [dic["row_names"]] here, since it is a matrix
                                pd.DataFrame(dic["data"], columns=dic["column_names"]).to_dict('records'),
                                id={"type": 'edit_non_graphing_data_matrices', 'index': name},
                                columns=[{'name': f"{col}", 'id': col} for col in dic["column_names"]],
                                editable=True
                            ),
                        ]) for name, dic in non_graph_data_matrix.items()
                    ]
                ]),

                # tab for the environment parameters, like pH and Temperature. 
                dcc.Tab(label='Environment Parameters', children=[
                    html.Div([
                        html.H2(f"DataTable for Environment Parameters"),
                        html.H4(f"Note: Some prameters wont influence the simulation. For example, changing M wont affect the number of steps in the lysis process, but overall should ahve an immediate effect on the simulation."),
                        dash_table.DataTable(
                            pd.DataFrame([analysis.environment_data]).to_dict('records'),
                            id="environment_data",
                            columns=[{"name": col, "id": col} for col in analysis.environment_data.keys()],
                            editable=True
                        ),
                    ])
              ]), 

                # tab for the settings, like the solver type and the min/max step size.
                dcc.Tab(label='Settings', children=[
                    html.H4(["These settings affect various parts of the simulation. For example the solver type (RK23 or RK45 or DOP853), or to use the defualt selected options by the solver or a linspace of selected options."]),
                    html.Button("Save Settings", id="save_settings"),
                    html.Div([
                        html.H4(["Solver Type"]), # dropdown of the solver types
                        dcc.Dropdown(
                            options=[
                                {'label': 'RK45', 'value': 'RK45'},
                                {'label': 'RK23', 'value': 'RK23'},
                                {'label': 'DOP853', 'value': 'DOP853'}, 
                                {'label': 'Radau', 'value': 'Radau'}, 
                                {'label': 'BDF', 'value': 'BDF'},
                                {'label': 'LSODA', 'value': 'LSODA'},
                            ],
                            value=initial_settings['solver_type'],
                            id={'type': 'settings', 'index': 'solver_type'},
                        ),
                        html.H4(["t_eval option"]), # evaluation option
                        dcc.Checklist(
                            options=[
                                {'label': 'Use solver suggested t_values (checked) or your own t_eval (unchecked) which uses the simulation time start and end and number of steps', 'value': 'option1'},
                            ],
                            value=['option1'],
                            id={'type': 'settings', 'index': 't_eval_option'}
                        ),
                        html.H4(["Minimum Step Size"]), # minimum step size
                        dcc.Input(
                            id={'type': 'settings', 'index': 'min_step'}, 
                            type="number",
                            placeholder="Minimum Step",
                            value=initial_settings['min_step'],
                            required=True, 
                            min=0.000001, 
                            max=1
                        ),
                        html.H4(["Max Step Size"]), # maximum step size
                        dcc.Input(
                            id={'type': 'settings', 'index': 'max_step'}, 
                            type="number",
                            placeholder="Maximum Step",
                            value=initial_settings['max_step'],
                            required=True, 
                            min=0.000001,
                            max=1
                        ),
                        html.H4(["Cutoff value for small numbers"]), # cutoff value for small numbers
                        dcc.Input(
                            id={'type': 'settings', 'index': 'cutoff_value'}, 
                            type="number",
                            placeholder="Cutoff Value for small numbers",
                            value=initial_settings['cutoff_value'],
                            required=True
                        ),
                        html.H4(["Dense Output"]), # dense output option
                        dcc.Checklist(
                            options=[
                                {'label': 'Use Dense Output', 'value': 'dense_output'},
                            ],
                            value=[],
                            id={'type': 'settings', 'index': 'dense_output'}
                        ),
                        html.H4(["Relative and Absolute Tolerance"]), # relative and absolute tolerance
                        dcc.Input(
                            id={'type': 'settings', 'index': 'rtol'}, 
                            type="number",
                            placeholder="Relative Tolerance",
                            value=initial_settings['rtol'], 
                            required=True
                        ),
                        dcc.Input(
                            id={'type': 'settings', 'index': 'atol'}, 
                            type="number",
                            placeholder="Absolute Tolerance",
                            value=initial_settings['atol'], 
                            required=True
                        ),
                        html.H4(["Simulation Length Time"]), # simulation length time
                        dcc.Input(
                            id={'type': 'settings', 'index': 'simulation_length'}, 
                            type="number",
                            placeholder="Simulation Length in time",
                            value=initial_settings['simulation_length'], 
                            required=True
                        ),
                    ])
                ]),
            ]),

            html.Div(style={'margin': '60px'}),
            html.Hr(),
            html.Div(style={'margin': '60px'}),
            dcc.Tabs([
                dcc.Tab(label='Serial Transfer', children=[
                    html.H4(["Note: Using the serial transfer function will take the final iteration values of the current simulation as shown above and divide it by the value shown below. So if the final value for a bacteria is 100 and the serial transfer value is 10, the new simulation will start with a bacteria value of 10. There is a special case for nutrients where the new value is added to the value associated in the Graphing Data (Initial Conditions) section. So for nutrients, if the final value is 100 and the serial transfer value is 10, and the \"Initial\" Condition is 50, the new simulation will start with a nutrient value of 100/10 + 50 = 60. If the checkbox is selected, the unique process will also apply to the phages and bacteria. If the checkbox is not selected, the unique process will only apply to the resources/nutrients. Change the value below to 1 if oyu want to simply add phages/bacteria/resources without removing substances. "]),
                    html.H4(["Note 2: These settings are also used in Parameter Analysis, Initial Value, and Phase Portrait if the associated checkbox is selected"]),
                    html.H4(["Serial Transfer Dilution rate: "]),
                    dcc.Input(
                        id="serial_transfer_value", type="number", placeholder="input with range",
                        min=1, max=1_000_000, step=0.01, value=10
                    ),
                    html.Br(),
                    html.H4(["Option to add phages and uninfected bacteria to serial transfer, uses value in Initial Condition"]),
                    dcc.Checklist(
                        options=[
                            {'label': 'Add Phages and Bacteria', 'value': 'option1'},
                        ],
                        value=[],
                        id='serial_transfer_bp_option'
                    ),
                    html.Br(),
                    html.H4(["Number of times to automatically run serial transfer"]),
                    dcc.Input(
                        id="serial_transfer_frequency",
                        type="number",
                        placeholder="1",
                        value="1"
                    ),
                    html.Br(),
                    dcc.Checklist(
                        options=[
                            {'label': 'Use log y axis (checked) or linear y axis (unchecked)', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='serial_transfier_figure_log_axis'
                    ),
                    html.Button("Run Serial Transfer", id="run_serial_transfer"),
                    html.H4(["The plots above in the Line Chart section will update with the new values after the serial transfer is complete, ensure that a model ahs already been run before running the serial transfer. To reset the chart, run the 'Rerun model' button above."]),
                    html.Div(style={'margin': '60px'}),
                ]),

                dcc.Tab(label='Parameter Analysis', children=[
                    html.H4(["Note: Choose 2 parameters of choice. Input the values you want to test separated by commas. The program will run the simulation for each combination of the two parameters and display the results in a heatmap. The heatmap represents the end value of the simulation for each combination of the two parameters. Make sure you choose an appropriate range of values to test and end simulation lenght before everything drops to 0!"]),
                    html.H4(["Choose two parameters to analyze"]),
                    dcc.Dropdown(both_params, id='parameter_analysis_param_name_1', value = both_params[0] if len(both_params) > 0 else None),
                    dcc.Dropdown(both_params, id='parameter_analysis_param_name_2', value = both_params[1] if len(both_params) > 1 else None),
                    html.H4(["Option 1: Input the values you want to test separated by commas"]),
                    dcc.Input(
                        id="parameter_analysis_input_1", 
                        type="text",
                        placeholder="Parameter 1 values to test",
                        value="0.01, 0.1, 1, 5, 50"
                    ),
                    dcc.Input(
                        id="parameter_analysis_input_2", 
                        type="text",
                        placeholder="Parameter 2 values to test",
                        value="0.02, 0.2, 2, 6, 10, 20, 50"
                    ),
                    html.Br(),
                    html.H4(["Option 2: Choose a start value and end value for each parameter separated by a '-' sign"]),
                    dcc.Input(
                        id="parameter_analysis_range_1",
                        type="text",
                        placeholder="0.01-0.8",
                    ), 
                    dcc.Input(
                        id="parameter_analysis_range_2",
                        type="text",
                        placeholder="0.01-0.8",
                    ), 
                    html.Br(),
                    html.H4(["And choose the values to test between the two values for uniform spaced intervals (including the end values)"]),
                    dcc.Input(
                        id="parameter_analysis_steps_1",
                        type="number",
                        placeholder="10",
                    ),
                    dcc.Input(
                        id="parameter_analysis_steps_2",
                        type="number",
                        placeholder="10",
                    ),
                    html.Br(),
                    html.Div(style={'margin': '60px'}),
                    dcc.Checklist(
                        options=[
                            {'label': 'Checked for running Option 1, unchecked for Option 2', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='parameter_analysis_option'
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'Use Serial Transfer', 'value': 'option1'},
                        ],
                        value=[],
                        id='parameter_analysis_use_serial_transfer'
                    ),
                    html.Button("Run Parameter Analysis", id="run_parameter_analysis"),
                    dcc.Slider(
                        min=0,
                        max=0,
                        value=0,
                        step=0.001,
                        id='parameter_analysis_slider',
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'Extrapolate value in case slider value is in between 2 calculated time intervals', 'value': 'option1'},
                        ],
                        value=[],
                        id='parameter_analysis_extrapolate'
                    ),
                    html.Div(children = [
                        *[
                            dcc.Graph(id={"type": "plot_parameter_analysis", "index": name}, style={'display': 'inline-block'}) for name in graph_data.keys()
                        ], 
                        dcc.Graph(id={"type": "plot_parameter_analysis", "index": "plot_parameter_analysis_bacteria_sum"}, style={'display': 'inline-block'}),
                    ],style={'margin': '60px'}), 
                ]),
                
                # tab for the initial value analysis, which is similar to the parameter analysis but for a single parameter
                dcc.Tab(label='Initial Value Analysis', children=[
                    html.H4(["Note: Choose a parameter of choice. Input the values you want to test separated by commas, or use a uniform seperated list. The program will run the simulation for each initial value and display the results on a graph."]),
                    dcc.Dropdown(both_params, id='initial_value_analysis_param_name', value = both_params[0] if len(both_params) > 0 else None),
                    html.H4(["Option 1: Input the values you want to test separated by commas"]),
                    dcc.Input(
                        id="initial_value_analysis_input", 
                        type="text",
                        placeholder="Initial values to test",
                        value="1, 10, 100, 1000"
                    ),
                    html.Br(),
                    html.H4(["Option 2: Choose a start value and end value for each parameter separated by a '-' sign"]),
                    dcc.Input(
                        id="initial_value_analysis_range",
                        type="text",
                        placeholder="1-100",
                    ),
                    html.Br(),
                    html.H4(["And choose the number of uniformly spaced values to test (including the end values)"]),
                    dcc.Input(
                        id="initial_value_analysis_steps",
                        type="number",
                        placeholder="10",
                    ),
                    html.Br(),
                    html.Div(style={'margin': '60px'}),
                    dcc.Checklist(
                        options=[
                            {'label': 'Checked for running Option 1, unchecked for Option 2', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='initial_value_analysis_option'
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'Use Serial Transfer', 'value': 'option1'},
                        ],
                        value=[],
                        id='initial_value_analysis_use_serial_transfer'
                    ),
                    html.H4(["Name for the run (optional, default = 'Run [number]):"]),
                    dcc.Input(
                        value="",
                        id='initial_value_analysis_run_name'
                    ),
                    html.H4(["Choose a scale for the graph (linear or log graph)"]),
                    dcc.Dropdown(['log-linear (log)', 'linear-linear (linear)'], id='initial_value_analysis_graph_scale', value = 'log-linear (log)'),
                    dcc.Input(
                        value="0.95",
                        id='initial_value_analysis_offset',
                        min=0.01,
                        max=1,
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'Log y axis IVA plot', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='initial_value_analysis_log_axis'
                    ),
                    html.Button("Run Initial Value Analysis", id="run_initial_value_analysis"),
                    html.H4(["Clear Bar Chart"]),
                    html.Button("Clear Bar Chart", id="clear_bar_chart"),
                    html.Div(style={'margin': '60px'}),
                    *[
                        dcc.Graph(id={"type": "plot_initial_value_analysis", "index": name}) for name in graph_data.keys()
                    ],
                    dcc.Graph(id={"type": "plot_initial_value_analysis", "index": "plot_initial_value_analysis_bacteria_sum"})
                ]),

                dcc.Tab(label='Phase Portrait', children=[
                    html.H4(["Note: Choose 2 parameters of choice. The program will run a simulation and plot a phase portrait of the two parameters. The phase portrait will show the relationship between the two parameters over time."]),
                    dcc.Dropdown(graph_data_name_list, id='phase_portrait_param_name_1', value = graph_data_name_list[0] if len(graph_data_name_list) > 0 else None),
                    dcc.Dropdown(graph_data_name_list, id='phase_portrait_param_name_2', value = graph_data_name_list[1] if len(graph_data_name_list) > 1 else None),
                    # TODO: remove the value from the input after testing
                    html.H4(["Option 1: Input the values you want to test separated by commas"]),
                    dcc.Input(
                        id="phase_portrait_starting_x", 
                        type="text",
                        placeholder="List of x starting values separated by commas",
                        value="48, 49, 50"
                    ),
                    dcc.Input(
                        id="phase_portrait_starting_y", 
                        type="text",
                        placeholder="List of y starting values separated by comma",
                        value="10, 50, 100"
                    ),
                    html.H4(["Option 2: Choose a start value and end value for each parameter separated by a '-' sign"]),
                    html.H4(["Parameter 1:"]),
                    dcc.Input(
                        id="phase_portrait_range_1", 
                        type="text",
                        placeholder="Start and end values for parameter 1 separated by a '-' sign",
                        value="48-50"
                    ),
                    dcc.Input(
                        id="phase_portrait_steps_1", 
                        type="number",
                        placeholder="Number of steps for parameter 1",
                        value="15"
                    ),
                    html.Br(),
                    html.H4(["Parameter 2:"]),
                    dcc.Input(
                        id="phase_portrait_range_2", 
                        type="text",
                        placeholder="Start and end values for parameter 2 separated by a '-' sign",
                        value="0.01-60"
                    ),
                    dcc.Input(
                        id="phase_portrait_steps_2", 
                        type="number",
                        placeholder="Number of steps for parameter 2",
                        value="15"
                    ),
                    html.Br(),
                    dcc.Checklist(
                        options=[
                            {'label': 'Use option 1 (checked) or option 2 (unchecked)', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='phase_portrait_auto_calculate_range'
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'Log x graph', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='phase_portrait_log_x'
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'Log y graph', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='phase_portrait_log_y'
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'Use Serial Transfer', 'value': 'option1'},
                        ],
                        value=[],
                        id='phase_portrait_use_serial_transfer'
                    ),
                    html.Button("Run Phase Portrait", id="run_phase_portrait"),
                    html.Div(style={'margin': '60px'}),
                    dcc.Graph(id="plot_phase_portrait")
                ]),

                dcc.Tab(label='SOBOL Analysis', children=[
                    html.H4("Runs a Sobol Analysis"),
                    *[
                        html.Div([
                            html.H4(name),
                            dcc.Input(
                                id={"type": "sobol_analysis_input", "index": name}, 
                                type="text",
                                placeholder="Sobol Analysis Input",
                                value="0.01-10"
                            )
                        ]) for name in both_params
                    ],
                    html.H4(['Number of samples, 2^x, where x is the number inputed below']),
                    dcc.Input(
                        id="SOBOL_analysis_samples", 
                        type="number",
                        placeholder="Number of samples",
                        value=2
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'Calculate 2nd order', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='SOBOL_analysis_2nd_order'
                    ),
                    dcc.Checklist(
                        options=[
                            {'label': 'Use Serial Transfer', 'value': 'option1'},
                        ],
                        value=[],
                        id='SOBOL_analysis_use_serial_transfer'
                    ),
                    dcc.Graph(id="SOBOL_analysis_parameter"),
                    dcc.Graph(id="SOBOL_analysis_time"), 
                    html.Button("Run SOBOL Analysis", id="run_SOBOL_analysis"),
                ]),
            ]),
        ])