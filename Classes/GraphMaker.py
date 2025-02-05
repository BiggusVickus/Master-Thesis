from tkinter import * 
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import simpledialog, messagebox
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(69)

class GraphMaker:
    def __init__(self, GUI):
        self.GUI = GUI
        self.graph = nx.Graph()

    def plot(self): 
        if not self.GUI:
            return
        for widget in self.window.winfo_children():
            if not isinstance(widget, Button):
                widget.destroy()
        fig = Figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.clear()
        pos = nx.spring_layout(self.graph, seed=69)
        color_map = []
        for node, attr in self.graph.nodes(data=True):
            if attr["node_type"] == "P":
                color_map.append('yellow')
            elif attr["node_type"] == "R":
                color_map.append('red')
            elif attr["node_type"] == "E":
                color_map.append('green')
            elif attr["node_type"] == "B":
                color_map.append('orange')
            else:
                continue
        nx.draw(self.graph, node_color=color_map, ax=ax, with_labels=True)
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas, self.window)
        toolbar.update()
        canvas.get_tk_widget().pack()

    def default_environment_data(self):
        string = ""
        string += f"Temperature: {self.randomize_parameter_value(25)}\n"
        string += f"pH: {self.randomize_parameter_value(7)}\n"
        string += f"Simulation_Length: {self.randomize_parameter_value(100)}\n"
        string += f"Time_Step: {self.randomize_parameter_value(0.1)}\n"
        string += f"Cutoff: {self.randomize_parameter_value(0.000001)}\n"
        return string

    def default_phage_data(self):
        string = ""
        string += f'Initial_Population: {self.randomize_parameter_value(100)}\n'
        string += f'Washout_Rate: {self.randomize_parameter_value(0.1)}\n'
        return string

    def default_bacteria_data(self):
        string = ""
        string += f"Initial_Population: {self.randomize_parameter_value(100)}\n"
        string += f"Growth_Rate: {self.randomize_parameter_value(0.1)}\n"
        string += f"Death_Rate: {self.randomize_parameter_value(0.1)}\n"
        string += f"Minimal_Temperature: {self.randomize_parameter_value(20)}\n"
        string += f"Optimal_Temperature: {self.randomize_parameter_value(25)}\n"
        string += f"Maximal_Temperature: {self.randomize_parameter_value(30)}\n"
        string += f"Minimal_pH: {self.randomize_parameter_value(6)}\n"
        string += f"Optimal_pH: {self.randomize_parameter_value(7)}\n"
        string += f"Maximal_pH: {self.randomize_parameter_value(8)}\n"
        return string

    def default_resource_data(self):
        string = ""
        string += f"Initial_Concentration: {self.randomize_parameter_value(100)}\n"
        string += f"Decay_Rate: {self.randomize_parameter_value(0.1)}\n"
        string += f"Replenishment_Rate: {self.randomize_parameter_value(0.1)}\n"
        return string

    def randomize_parameter_value(self, main_value, sigma = 1):
        return np.random.normal(main_value, sigma)

    def default_p_b_data(self):
        string = ""
        string += f"Burst_Size: {self.randomize_parameter_value(100)}\n"
        string += f"Adsorption_Rate_Phage_to_Bacteria: {self.randomize_parameter_value(0.1)}\n"
        string += f"Adsorption_Rate_Bacteria_to_Phage: {self.randomize_parameter_value(0.1)}\n"
        string += f"Lysis_Rate: {self.randomize_parameter_value(0.1)}\n"
        string += f"Carying_Capacity: {self.randomize_parameter_value(100)}\n"
        string += f"r: {self.randomize_parameter_value(0.1)}\n"
        string += f"tau: {self.randomize_parameter_value(0.5)}\n"
        return string

    def default_b_r_data(self):
        string = ""
        string += f"Uptake_Rate: {self.randomize_parameter_value(0.1)}\n"
        string += f"Release_Rate: {self.randomize_parameter_value(0.1)}\n"
        string += f"e: {self.randomize_parameter_value(0.1)}\n"
        string += f"v: {self.randomize_parameter_value(0.1)}\n"
        string += f"K: {self.randomize_parameter_value(100)}\n"
        return string

    def add_node(self, node_type, node_name, node_data = None):
        if node_type == None and node_name == None:
            return self.error_message("Node name and type cannot be empty")
        node_type = node_type.strip()
        node_name = node_name.strip()
        if node_type not in ["P", "B", "R"]:
            return self.error_message("Node type must be P, B, or R")
        for node in self.graph.nodes:
            if node_name == node:
                return self.error_message("Node name already exists")
        
        self.graph.add_node(node_name)
        self.graph.nodes[node_name]['node_type'] = node_type
        if node_data is None:
            node_data = {
            "E": self.default_environment_data,
            "P": self.default_phage_data,
            "B": self.default_bacteria_data,
            "R": self.default_resource_data
            }.get(node_type, lambda: None)()

        self.graph.nodes[node_name]['data'] = node_data
        self.plot()
        
    def remove_node(self, node_name):
        node_name = node_name.strip()
        if node_name in self.graph:
            self.graph.remove_node(node_name)
            self.plot()
        else:
            return self.error_message("Node not found")

    def add_edge(self, node1 = None, node2 = None, edge_data = None):
        if node1 == None or node2 == None:
            return self.error_message("Node names cannot be empty")
        node1 = node1.strip()
        node2 = node2.strip()
        if self.verify_edge_connections(node1, node2, "E", "E"):
            return self.error_message("Cannot connect E node to any other node")
        if node1 not in self.graph or node2 not in self.graph:
            return self.error_message("One or both nodes not found")
        self.graph.add_edge(node1, node2)
        if self.verify_edge_connections(node1, node2, "P", "P"):
            node_data = "Default edge data"
        elif self.verify_edge_connections(node1, node2, "P", "B"):
            node_data = self.default_p_b_data()
        elif self.verify_edge_connections(node1, node2, "P", "R"):
            node_data = "Default edge data"
        elif self.verify_edge_connections(node1, node2, "B", "B"):
            node_data = "Default edge data"
        elif self.verify_edge_connections(node1, node2, "B", "R"):
            node_data = self.default_b_r_data()
        elif self.verify_edge_connections(node1, node2, "R", "R"):
            node_data = "Default edge data"
        else:
            node_data = "Default edge data"
        self.graph.edges[node1, node2]['data'] = node_data
        self.plot()

    def remove_edge(self, node1 = None, node2 = None):
        if node1 == None or node2 == None:
            return self.error_message("Node names cannot be empty")
        node1 = node1.strip()
        node2 = node2.strip()
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)
            self.plot()
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
            self.graph.add_node("P" + str(i))
            self.graph.nodes["P" + str(i)]['node_type'] = "P"
            self.graph.nodes["P" + str(i)]['data'] = self.default_phage_data()
        for i in range(B):
            self.graph.add_node("B" + str(i))
            self.graph.nodes["B" + str(i)]['node_type'] = "B"
            self.graph.nodes["B" + str(i)]['data'] = self.default_bacteria_data()
        for i in range(R):
            self.graph.add_node("R" + str(i))
            self.graph.nodes["R" + str(i)]['node_type'] = "R"
            self.graph.nodes["R" + str(i)]['data'] = self.default_resource_data()
        self.plot()

    def verify_edge_connections(self, node1, node2, type1, type2):
        return (self.graph.nodes[node1]['node_type'] == type1 and self.graph.nodes[node2]['node_type'] == type2) or (self.graph.nodes[node1]['node_type'] == type2 and self.graph.nodes[node2]['node_type'] == type1)
    
    def mass_create_edges(self, edge_connections):
        # Function to handle the "Submit" button click
        error_text = ""
        for edge in edge_connections:
            node1 = edge[0].strip()
            node2 = edge[1].strip()
            if (len(node1) == 3):
                node_data = edge[2]
            else:
                if self.verify_edge_connections(node1, node2, "P", "P"):
                    node_data = "Default edge data"
                elif self.verify_edge_connections(node1, node2, "P", "B"):
                    node_data = self.default_p_b_data()
                elif self.verify_edge_connections(node1, node2, "P", "R"):
                    node_data = "Default edge data"
                elif self.verify_edge_connections(node1, node2, "B", "B"):
                    node_data = "Default edge data"
                elif self.verify_edge_connections(node1, node2, "B", "R"):
                    node_data = self.default_b_r_data()
                elif self.verify_edge_connections(node1, node2, "R", "R"):
                    node_data = "Default edge data"
                else:
                    node_data = "Default edge data"
            if node1 == None or node2 == None:
                error_text += f"Node name {node1} and/or {node2} cannot be empty"
            if (self.verify_edge_connections(node1, node2, "E", "E")):
                    error_text += "Cannot connect any node to an E node\n"
            if node1 not in self.graph or node2 not in self.graph:
                error_text += f"{node1} or {node2} not found in graph\n"
            self.graph.add_edge(node1, node2)
            self.graph.edges[node1, node2]['data'] = node_data
            self.plot()
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
        try:
            if not file_name.endswith(('.gexf', '.gml', '.graphml', '.net')):
                return self.error_message("File must be of type .gexf, .gml, .graphml, or .net")
            if (file_name.endswith('.gexf')):
                nx.read_gexf(self.graph, file_name)
            elif (file_name.endswith('.gml')):
                nx.read_gml(self.graph, file_name)
            elif (file_name.endswith('.graphml')):
                nx.read_graphml(self.graph, file_name)
            elif (file_name.endswith('.net')):
                nx.read_pajek(self.graph, file_name)
        except:
            return self.error_message("File not found")
        self.plot()

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
        self.graph.edges[node1, node2]['data'] = user_input

    def error_message(self, message, throw_error = True, message_type = "Error"):
        if (self.GUI):
            messagebox.showerror(message_type, message)
        else:
            if (throw_error):
                return Exception(message)
            else:
                print(message)
                return
