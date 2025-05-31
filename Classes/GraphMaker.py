from tkinter import * 
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import messagebox
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time

class GraphMaker:
    def __init__(self, GUI, seed=None):
        self.GUI = GUI
        self.graph = nx.MultiGraph()
        if seed is not None:
            np.random.seed(seed)
        else:
            seed = int(1000 * time.time())% 2**32
            np.random.seed(seed)
        self.seed = seed

    def plot(self): 
        if not self.GUI:
            return
        for widget in self.window.winfo_children():
            if not isinstance(widget, Button):
                widget.destroy()
        fig = Figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)
        ax.clear()
        pos = nx.multipartite_layout(self.graph)
        color_map = []
        for node, attr in self.graph.nodes(data=True):
            if attr["node_type"] == "P":
                color_map.append("#83C9E1")
            elif attr["node_type"] == "R":
                color_map.append("#FF9C46")
            elif attr["node_type"] == "B":
                color_map.append("#FF96FD")
            elif attr["node_type"] == "S" or attr["node_type"] == "E":
                color_map.append("#7CEC7C")
            else:
                color_map.append('yellow')
        nx.draw_networkx_nodes(self.graph, pos, node_color=color_map, ax=ax)
        straight_edges = []
        curved_edges = []
        if any(u == v for u, v in self.graph.edges()):
            # If there are self-loops, use keys=True to handle MultiGraph edges
            for u, v, keys in self.graph.edges(keys=True):
                if u == v:
                    curved_edges.append((u, v))
                elif self.graph.nodes[u]["subset"] == self.graph.nodes[v]["subset"]:
                    curved_edges.append((u, v))
                else:
                    straight_edges.append((u, v))
        else:
            # No self-loops, keys not needed
            for u, v in self.graph.edges():
                if self.graph.nodes[u]["subset"] == self.graph.nodes[v]["subset"]:
                    curved_edges.append((u, v))
                else:
                    straight_edges.append((u, v))
        nx.draw_networkx_edges(self.graph, pos, edgelist=straight_edges, edge_color='black', ax=ax)
        nx.draw_networkx_edges(self.graph, pos, edgelist=curved_edges, edge_color='red', connectionstyle="arc3,rad=0.6", ax=ax)
        nx.draw_networkx_labels(self.graph, pos, ax=ax)
        
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def default_environment_data(self):
        string = ""
        string += f"M: 4\n"
        string += f"washout: 0\n"
        return string
    
    def default_settings_data(self):
        string = ""
        string += f"solver_type: RK45\n"
        string += f"t_eval_option: True\n"
        string += f"t_eval_steps: 200\n"
        string += f"min_step: 0.01\n"
        string += f"max_step: 0.1\n"
        string += f"cutoff_value: 0.000001\n"
        string += f"dense_output: False\n"
        string += f"rtol: 0.001\n"
        string += f"atol: 0.000001\n"
        string += f"t_start: 0\n"
        string += f"simulation_length: 15\n"
        return string

    def default_phage_data(self):
        string = ""
        string += f'Initial_Population: {np.random.randint(5, 15)}\n'
        return string

    def default_bacteria_data(self):
        string = ""
        string += f"Initial_Population: {np.random.randint(50, 100)}\n"
        string += f"tau: {np.random.uniform(0.7, 3.1)}\n"
        return string

    def default_resource_data(self):
        string = ""
        string += f"Initial_Concentration: {np.random.randint(200, 400)}\n"
        string += f"washin: 0\n"
        return string
    
    def default_node_data(self):
        return ""

    def default_p_p_data(self):
        return ""
    
    def default_p_b_data(self):
        string = ""
        string += f"Burst_Size: {np.random.randint(10, 60)}\n"
        string += f"r: {np.random.uniform(0.001, 0.15)}\n"
        return string

    def default_p_r_data(self):
        return ""

    def default_b_b_data(self):
        return ""

    def default_b_r_data(self):
        string = ""
        string += f"v: {np.random.uniform(0.8, 1.7)}\n"
        string += f"e: {np.random.uniform(0.1, 0.2)}\n"
        string += f"K: {np.random.uniform(10, 150)}\n"
        return string
    
    def default_r_r_data(self):
        return ""

    def default_edge_data(self):
        return ""

    def add_node_to_graph(self, node_type, node_name, node_data = None):
        if node_type == None or node_name == None or node_type == "" or node_name == "":
            return self.error_message("Node name and type cannot be empty")
        if node_type == "E":
            for node, attr in self.graph.nodes(data=True):
                if attr["node_type"] == "E":
                    return self.error_message("Node of type 'E' (Environment node) already exists")
        if node_type == "S":
            for node, attr in self.graph.nodes(data=True):
                if attr["node_type"] == "S":
                    return self.error_message("Node of type 'S' (Settings node) already exists")
        node_type = node_type.strip()
        node_name = node_name.strip()
        for node in self.graph.nodes:
            if node_name == node:
                return self.error_message("Node name already exists")
        
        self.graph.add_node(node_name)
        self.graph.nodes[node_name]['node_type'] = node_type
        if node_data is None:
            node_data = {
            "E": self.default_environment_data,
            "S": self.default_settings_data,
            "P": self.default_phage_data,
            "B": self.default_bacteria_data,
            "R": self.default_resource_data
            }.get(node_type, lambda: None)()
        if node_type == "P":
            self.graph.nodes[node_name]['subset'] = "0"
        elif node_type == "B":
            self.graph.nodes[node_name]['subset'] = "1"
        elif node_type == "R":
            self.graph.nodes[node_name]['subset'] = "2"
        elif node_type == "E" or node_type == "S":
            self.graph.nodes[node_name]['subset'] = "4"
        else:
            self.graph.nodes[node_name]['subset'] = "3"

        self.graph.nodes[node_name]['data'] = node_data
        
    def remove_node(self, node_name):
        node_name = node_name.strip()
        if node_name in self.graph:
            self.graph.remove_node(node_name)
        else:
            return self.error_message("Node not found")

    def add_edge(self, node1 = None, node2 = None, edge_data = None):
        if node1 == None or node2 == None:
            return self.error_message("Node names cannot be empty")
        node1 = node1.strip()
        node2 = node2.strip()
        if node1 == "E" or node2 == "E" or node1 == "S" or node2 == "S":
            return self.error_message("Cannot connect E or S node to any other node")
        if node1 not in self.graph or node2 not in self.graph:
            return self.error_message("One or both nodes not found")
        if self.graph.has_edge(node1, node2):
            return self.error_message("Edge already exists")
        self.graph.add_edge(node1, node2)
        if self.verify_edge_connections(node1, node2, "P", "P"):
            node_data = self.default_p_p_data()
        elif self.verify_edge_connections(node1, node2, "P", "B"):
            node_data = self.default_p_b_data()
        elif self.verify_edge_connections(node1, node2, "P", "R"):
            node_data = self.default_p_r_data()
        elif self.verify_edge_connections(node1, node2, "B", "B"):
            node_data = self.default_b_b_data()
        elif self.verify_edge_connections(node1, node2, "B", "R"):
            node_data = self.default_b_r_data()
        elif self.verify_edge_connections(node1, node2, "R", "R"):
            node_data = self.default_r_r_data()
        else:
            node_data = self.default_edge_data()
        self.graph.edges[node1, node2, 0]['data'] = node_data

    def remove_edge(self, node1 = None, node2 = None):
        if node1 == None or node2 == None:
            return self.error_message("Node names cannot be empty")
        node1 = node1.strip()
        node2 = node2.strip()
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)
        else:
            return self.error_message("Edge not found")

    def mass_create_nodes(self, P = 0, B = 0, R = 0):
        if type(P) != int or type(B) != int or type(R) != int:
            if not isinstance(P, int) or not isinstance(B, int) or not isinstance(R, int):
                return self.error_message("Number of nodes must be an integer")
            if P < 0 or B < 0 or R < 0:
                return self.error_message("Number of nodes must be greater than or equal to 0")
            return self.error_message("Number of nodes must be an integer")
        for i in range(P):
            self.add_node_to_graph("P", "P" + str(i))
        for i in range(B):
            self.add_node_to_graph("B", "B" + str(i))
        for i in range(R):
            self.add_node_to_graph("R", "R" + str(i))

    def verify_edge_connections(self, node1, node2, type1, type2):
        if type1 == None or type2 == None:
            return False
        if node1 == None or node2 == None:
            return False
        if node1 not in self.graph or node2 not in self.graph:
            return False
        if type1 == "E" or type2 == "E" or type1 == "S" or type2 == "S":
            return False
        return (self.graph.nodes[node1]['node_type'] == type1 and self.graph.nodes[node2]['node_type'] == type2) or (self.graph.nodes[node1]['node_type'] == type2 and self.graph.nodes[node2]['node_type'] == type1)
    
    def mass_create_edges(self, tuple_of_edges:tuple, edge_data = None):
        error_text = ""
        for i, target in enumerate(tuple_of_edges):
            if len(target) != 2:
                error_text += "Each tuple must have 2 elements, skipping edge\n"
                continue
            node1 = target[0].strip()
            node2 = target[1].strip()
            if (edge_data is not None):
                if len(edge_data) == 1 or isinstance(edge_data, str):
                    if isinstance(edge_data, str):
                        edge_data = edge_data
                    else: 
                        edge_data = edge_data[0]
                elif len(tuple_of_edges) != len(edge_data):
                    return self.error_message("Number of target nodes must match number of edge data")
                else:
                    edge_data = edge_data[i]
            if node1 == None or node2 == None:
                error_text += f"Node name {node1} and/or {node2} cannot be empty"
                continue
            if (self.verify_edge_connections(node1, node2, "E", "E")):
                error_text += "Cannot connect any node to an E node\n"
                continue
            if node1 not in self.graph or node2 not in self.graph:
                error_text += f"{node1} or {node2} not found in graph\n"
                continue
            if self.graph.has_edge(node1, node2):
                error_text += f"Edge between {node1} and {node2} already exists, skipping\n"
                continue
            self.add_edge(node1, node2, edge_data)
        return error_text

    def export_graph_to_file(self, file_name:str =None):
        if file_name == None or file_name == "":
            return self.error_message("No file name provided")
        if not file_name.endswith(('.gexf', '.gml', '.graphml', '.net')):
            return self.error_message("File must be of type .gexf, .gml, .graphml, or .net")
        if (file_name.endswith('.gexf')):
            nx.write_gexf(self.graph, file_name)
        elif (file_name.endswith('.gml')):
            nx.write_gml(self.graph, file_name)
        elif (file_name.endswith('.graphml')):
            nx.write_graphml(self.graph, file_name)
        elif (file_name.endswith('.net')):
            nx.write_pajek(self.graph, file_name)

    def import_graph_from_file(self, file_name:str=None):
        if file_name == None:
            return self.error_message("No file name provided")
        if not file_name.endswith(('.gexf', '.gml', '.graphml', '.net')):
            return self.error_message("File must be of type .gexf, .gml, .graphml, or .net")
        try:
            if (file_name.endswith('.gexf')):
                self.graph = nx.read_gexf(file_name)
            elif (file_name.endswith('.gml')):
                self.graph = nx.read_gml(file_name)
            elif (file_name.endswith('.graphml')):
                self.graph = nx.read_graphml(file_name)
            elif (file_name.endswith('.net')):
                self.graph = nx.read_pajek(file_name)
        except Exception as e:
            return self.error_message(str(e))

    def edit_node_attributes(self, node_name:str, node_data):
        if node_name is None or node_name == "" or node_data is None:
            return self.error_message("Node name cannot be empty")
        if node_name not in self.graph:
            return self.error_message("Node name not found")
        if node_data is None:
            return self.error_message("Node data cannot be empty")
        node_name = node_name.strip()
        self.graph.nodes[node_name]['data'] = node_data

    def edit_edge_attributes(self, node1:str, node2:str, user_input):
        if node1 not in self.graph or node2 not in self.graph:
            return self.error_message("Node name not found")
        node1 = node1.strip()
        node2 = node2.strip()
        edge_attributes = self.graph[node1][node2]
        edge_attributes[0]['data'] = user_input

    def error_message(self, message, throw_error = True, message_type = "Error"):
        if (self.GUI):
            messagebox.showerror(message_type, message)
        else:
            if (throw_error):
                return Exception(message)
            else:
                print(message)
                return

    def get_graph_object(self):
        return self.graph

    def set_graph_object(self, graph):
        self.graph = graph