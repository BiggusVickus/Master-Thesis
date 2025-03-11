from dash import Dash, dash_table, html, Input, Output, callback, ALL, State
from dash import dcc
from collections import OrderedDict
import pandas as pd

def html_code(graph_data, non_graph_data_vector, non_graph_data_matrix, graph):
    graph_data_name_list = [name for name in graph_data.keys()]
    non_graph_data_name_list = [name for name in OrderedDict(list(non_graph_data_vector.items()) + list(non_graph_data_matrix.items()))]
    both_params = graph_data_name_list + non_graph_data_name_list
    return html.Div([
            html.H1("Line Chart"),
            *[
                dcc.Graph(id={"type": "plot_basic_graph_data", "index": name}) for name in graph_data.keys()
            ],
            html.Div(style={'margin': '60px'}),
            html.Hr(),
            html.Div(style={'margin': '60px'}),
            html.Button('Save and rerun model', id='run_basic_model'),
            dcc.Tabs([
                dcc.Tab(label='Graphing Data (Initial Conditions)', children=[
                    *[
                        html.Div([
                            html.H2(f"DataTable for {name}"),
                            html.H3(f"Row names {dic['row_names']}") if dic["row_names"] is not None else None,
                            dash_table.DataTable(
                                pd.DataFrame([dic["data"]], columns=dic["column_names"]).to_dict('records') if dic["row_names"] is None else pd.DataFrame(dic["data"], columns=dic["column_names"]).to_dict('records'),
                                id={"type": 'edit_graphing_data', 'index': name},
                                columns=[{'name': f"{col}", 'id': col} for col in dic["column_names"]],
                                editable=True
                            ),
                        ]) for name, dic in graph_data.items()
                    ]
                ]),
                dcc.Tab(label='Non Graphing Data (Parameter Values): Vectors', children=[
                    *[
                        html.Div([
                            html.H2(f"DataTable for {name}"),
                            dash_table.DataTable(
                                pd.DataFrame([dic["data"]], columns=dic["column_names"]).to_dict('records'),
                                id={"type": 'edit_non_graphing_data_vectors', 'index': name},
                                columns=[{'name': f"{col}", 'id': col} for col in dic["column_names"]],
                                editable=True
                            ),
                        ]) for name, dic in non_graph_data_vector.items()
                    ]
                ]),
                dcc.Tab(label='Non Graphing Data (Parameter Values): Matrices', children=[
                    *[
                        html.Div([
                            html.H2(f"DataTable for {name}"),
                            html.H3(f"Row Names: {dic['row_names']}"),
                            dash_table.DataTable(
                                pd.DataFrame(dic["data"], columns=dic["column_names"]).to_dict('records'),
                                id={"type": 'edit_non_graphing_data_matrices', 'index': name},
                                columns=[{'name': f"{col}", 'id': col} for col in dic["column_names"]],
                                editable=True
                            ),
                        ]) for name, dic in non_graph_data_matrix.items()
                    ]
                ]),
                dcc.Tab(label='Environment Parameters', children=[
                    html.Div([
                        html.H2(f"DataTable for Environment Parameters"),
                        html.H4(f"Note: Some prameters wont influence the simulation. For example, changing M wont affect the number of steps in the lysis process, but overall should ahve an immediate effect on the simulation."),
                        dash_table.DataTable(
                            pd.DataFrame([graph.environment_node_data]).to_dict('records'),
                            id="environment_data",
                            columns=[{"name": col, "id": col} for col in graph.environment_node_data.keys()],
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
                        id='serial_tranfer_bp_option'
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
                    html.Button("Run Serial Transfer", id="run_serial_transfer"),
                    html.Div(style={'margin': '60px'}),
                ]),

                dcc.Tab(label='Parameter Analysis', children=[
                    html.H4(["Note: Choose 2 parameters of choice. Input the values you want to test separated by commas. The program will run the simulation for each combination of the two parameters and display the results in a heatmap. The heatmap represents the end value of the simulation for each combination of the two parameters. Make sure you choose an appropriate range of values to test and end simulation lenght before everything drops to 0!"]),
                    html.H4(["Choose two parameters to analyze"]),
                    dcc.Dropdown(both_params, id='parameter_analysis_param_name_1', value = both_params[0] if len(both_params) > 0 else None),
                    dcc.Dropdown(both_params, id='parameter_analysis_param_name_2', value = both_params[1] if len(both_params) > 1 else None),
                    dcc.Checklist(
                        options=[
                            {'label': 'Checked for running Option 1, unchecked for Option 2', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='parameter_analysis_option'
                    ),
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
                    dcc.Checklist(
                        options=[
                            {'label': 'Use Serial Transfer', 'value': 'option1'},
                        ],
                        value=[],
                        id='parameter_analysis_use_serial_transfer'
                    ),
                    html.Button("Run Parameter Analysis", id="run_parameter_analysis"),
                    html.Div(style={'margin': '60px'}),
                    *[
                        dcc.Graph(id={"type": "plot_parameter_analysis", "index": name}) for name in graph_data.keys()
                    ],
                ]),

                dcc.Tab(label='Initial Value Analysis', children=[
                    html.H4(["Note: Choose a parameter of choice. Input the values you want to test separated by commas, or use a uniform seperated list. The program will run the simulation for each initial value and display the results on a graph."]),
                    dcc.Dropdown(both_params, id='initial_value_analysis_param_name', value = both_params[0] if len(both_params) > 0 else None),
                    dcc.Checklist(
                        options=[
                            {'label': 'Checked for running Option 1, unchecked for Option 2', 'value': 'option1'},
                        ],
                        value=['option1'],
                        id='initial_value_analysis_option'
                    ),
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
                    html.H4(["And choose the values to test between the two values for a uniform distribution (including the end values)"]),
                    dcc.Input(
                        id="initial_value_analysis_steps",
                        type="number",
                        placeholder="10",
                    ),
                    html.Br(),
                    dcc.Checklist(
                        options=[
                            {'label': 'Use Serial Transfer', 'value': 'option1'},
                        ],
                        value=[],
                        id='initial_value_analysis_use_serial_transfer'
                    ),
                    html.Button("Run Initial Value Analysis", id="run_initial_value_analysis"),
                    html.Div(style={'margin': '60px'}),
                    *[
                        dcc.Graph(id={"type": "plot_initial_value_analysis", "index": name}) for name in graph_data.keys()
                    ],
                ]),

                dcc.Tab(label='Phase Portrait', children=[
                    html.H4(["Note: Choose 2 parameters of choice. The program will run a simulation and plot a phase portrait of the two parameters. The phase portrait will show the relationship between the two parameters over time."]),
                    dcc.Dropdown(graph_data_name_list, id='phase_portrait_param_name_1', value = graph_data_name_list[0] if len(graph_data_name_list) > 0 else None),
                    dcc.Dropdown(graph_data_name_list, id='phase_portrait_param_name_2', value = graph_data_name_list[1] if len(graph_data_name_list) > 1 else None),
                    # TODO: remove the value from the input after testing
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
                            {'label': 'Use Serial Transfer', 'value': 'option1'},
                        ],
                        value=[],
                        id='phase_portrait_use_serial_transfer'
                    ),
                    html.Button("Run Phase Portrait", id="run_phase_portrait"),
                    html.Div(style={'margin': '60px'}),
                    dcc.Graph(id="plot_phase_portrait")
                ]),
            ]),
        ])