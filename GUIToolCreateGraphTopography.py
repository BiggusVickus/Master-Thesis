from tkinter import * 
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
import tkinter as tk
from tkinter import Canvas, simpledialog, messagebox
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(69)

graph = nx.Graph()

def create_button(command, text, row = 0, column = 0):
    button = Button(master = window,
                     command = command, 
                     height = 1,
                     width = 10, 
                     text = text) 
    return button
    # button.grid(row=row, column=column)

def plot(): 
    for widget in window.winfo_children():
        if not isinstance(widget, Button):
            widget.destroy()
    fig = Figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)
    ax.clear()
    pos = nx.spring_layout(graph, seed=69)
    color_map = []
    for node, attr in graph.nodes(data=True):
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
    nx.draw(graph, node_color=color_map, ax=ax, with_labels=True)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack()

def default_environment_data():
    string = ""
    string += f"Temperature: {randomize_parameter_value(25)}\n"
    string += f"pH: {randomize_parameter_value(7)}\n"
    string += f"Simulation Time: {randomize_parameter_value(100)}\n"
    string += f"Time Step: {randomize_parameter_value(0.1)}\n"
    return string

def default_phage_data():
    string = ""
    string += f'Initial Population: {randomize_parameter_value(100)}\n'
    string += f'Washout Rate: {randomize_parameter_value(0.1)}\n'
    return string

def default_bacteria_data():
    string = ""
    string += f"Initial Population: {randomize_parameter_value(100)}\n"
    string += f"Growth Rate: {randomize_parameter_value(0.1)}\n"
    string += f"Death Rate: {randomize_parameter_value(0.1)}\n"
    string += f"Minimal Temperature: {randomize_parameter_value(20)}\n"
    string += f"Optimal Temperature: {randomize_parameter_value(25)}\n"
    string += f"Maximal Temperature: {randomize_parameter_value(30)}\n"
    string += f"Minimal pH: {randomize_parameter_value(6)}\n"
    string += f"Optimal pH: {randomize_parameter_value(7)}\n"
    string += f"Maximal pH: {randomize_parameter_value(8)}\n"
    return string

def default_resource_data():
    string = ""
    string += f"Initial Concentration: {randomize_parameter_value(100)}\n"
    string += f"Decay Rate: {randomize_parameter_value(0.1)}\n"
    string += f"Replenishment Rate: {randomize_parameter_value(0.1)}\n"
    return string

def randomize_parameter_value(main_value, sigma = 1):
    return np.random.normal(main_value, sigma)

def default_p_b_data():
    string = ""
    string += f"Burst Size: {randomize_parameter_value(100)}\n"
    string += f"Adsorption Rate Phage to Bacteria: {randomize_parameter_value(0.1)}\n"
    string += f"Adsorption Rate Bacteria to Phage: {randomize_parameter_value(0.1)}\n"
    string += f"Lysis Rate: {randomize_parameter_value(0.1)}\n"
    string += f"Carying Capacity: {randomize_parameter_value(100)}\n"
    string += f"r: {randomize_parameter_value(0.1)}\n"
    string += f"tau: {randomize_parameter_value(0.5)}\n"
    return string

def default_b_r_data():
    string = ""
    string += f"Uptake Rate: {randomize_parameter_value(0.1)}\n"
    string += f"Release Rate: {randomize_parameter_value(0.1)}\n"
    string += f"e: {randomize_parameter_value(0.1)}\n"
    string += f"v: {randomize_parameter_value(0.1)}\n"
    string += f"K: {randomize_parameter_value(100)}\n"

    return string

def add_node():
    node_type = simpledialog.askstring("Input", "Enter exactly P, B, R, E for type of input:")
    if node_type not in ["P", "B", "R", "E"]:
        messagebox.showerror("Error", "Invalid node type for Phage, Bacteria, Resource, Environment")
        return
    node_name = simpledialog.askstring("Input", "Enter node name:")
    for node in graph.nodes:
        if node_name == node:
            messagebox.showerror("Error", "Node already exists")
            return
    if node_name:
        graph.add_node(node_name)
        if node_type == "E":
            graph.nodes[node_name]['node_type'] = "E"
            graph.nodes[node_name]['data'] = default_environment_data()
        elif node_type == "P":
            graph.nodes[node_name]['node_type'] = "P"
            graph.nodes[node_name]['data'] = default_phage_data()
        elif node_type == "B":
            graph.nodes[node_name]['node_type'] = "B"
            graph.nodes[node_name]['data'] = default_bacteria_data()
        elif node_type == "R":
            graph.nodes[node_name]['node_type'] = "R"
            graph.nodes[node_name]['data'] = default_resource_data()
        plot()
    
def remove_node():
    node = simpledialog.askstring("Input", "Enter node name to remove:")
    if node in graph:
        graph.remove_node(node)
        plot()
    else:
        messagebox.showerror("Error", "Node not found")

def add_edge():
    node1 = simpledialog.askstring("Input", "Enter first node name:")
    node2 = simpledialog.askstring("Input", "Enter second node name:")
    if node1 == node2:
        messagebox.showerror("Error", "Cannot connect a node to itself")
        return
    if node1 == None or node2 == None:
        messagebox.showerror("Error", "Node names cannot be empty")
        return
    if (graph.nodes[node1]['node_type'] == "E") or (graph.nodes[node2]['node_type'] == "E"):
        messagebox.showerror("Error", "Cannot connect E node to any other node, it is an environment node and affects every P, B, R node")
        return

    if node1 in graph and node2 in graph:
        graph.add_edge(node1, node2)
        if (graph.nodes[node1]['node_type'] == "P" and graph.nodes[node2]['node_type'] == "B") or (graph.nodes[node1]['node_type'] == "B" and graph.nodes[node2]['node_type'] == "P"):
            graph.edges[node1, node2]['data'] = default_p_b_data()
        elif (graph.nodes[node1]['node_type'] == "B" and graph.nodes[node2]['node_type'] == "R") or (graph.nodes[node1]['node_type'] == "R" and graph.nodes[node2]['node_type'] == "B"):
            graph.edges[node1, node2]['data'] = default_b_r_data()
        else:
            graph.edges[node1, node2]['data'] = "Default edge data"
        plot()
    else:
        messagebox.showerror("Error", "One or both nodes not found")


def remove_edge():
    node1 = simpledialog.askstring("Input", "Enter first node name:")
    node2 = simpledialog.askstring("Input", "Enter second node name:")
    if graph.has_edge(node1, node2):
        graph.remove_edge(node1, node2)
        plot()
    else:
        messagebox.showerror("Error", "Edge not found")

def mass_create_nodes():
    P = simpledialog.askinteger("Input", "Enter number of P nodes:")
    B = simpledialog.askinteger("Input", "Enter number of B nodes:")
    R = simpledialog.askinteger("Input", "Enter number of R nodes:")
    E = simpledialog.askinteger("Input", "Enter number of E nodes:")
    for i in range(P):
        graph.add_node("P" + str(i))
        graph.nodes["P" + str(i)]['node_type'] = "P"
        graph.nodes["P" + str(i)]['data'] = default_phage_data()
    for i in range(B):
        graph.add_node("B" + str(i))
        graph.nodes["B" + str(i)]['node_type'] = "B"
        graph.nodes["B" + str(i)]['data'] = default_bacteria_data()
    for i in range(R):
        graph.add_node("R" + str(i))
        graph.nodes["R" + str(i)]['node_type'] = "R"
        graph.nodes["R" + str(i)]['data'] = default_resource_data()
    for i in range(E):
        graph.add_node("E" + str(i))
        graph.nodes["E" + str(i)]['node_type'] = "E"
        graph.nodes["E" + str(i)]['data'] = default_environment_data()
    plot()

def mass_create_edges():
    new_window = tk.Toplevel(window)
    new_window.title("New Window")

    # Create a Text widget for multiline input
    text_widget = tk.Text(new_window, height=5, width=30)
    text_widget.pack(pady=20)

    # Function to handle the "Submit" button click
    def submit():
        user_input = text_widget.get("1.0", tk.END) # Get all text from the widget
        lines = user_input.split("\n")
        for line in lines:
            if line:
                node1, node2 = line.split()
                if node1 == node2:
                    messagebox.showerror("Error", "Cannot connect a node to itself")
                    return
                if node1 == None or node2 == None:
                    messagebox.showerror("Error", "Node names cannot be empty")
                    return
                if (graph.nodes[node1]['node_type'] == "E") or (graph.nodes[node2]['node_type'] == "E"):
                    messagebox.showerror("Error", "Cannot connect E node to any other node, it is an environment node and affects every P, B, R node")
                    return

                if node1 in graph and node2 in graph:
                    graph.add_edge(node1, node2)
                    if (graph.nodes[node1]['node_type'] == "P" and graph.nodes[node2]['node_type'] == "B") or (graph.nodes[node1]['node_type'] == "B" and graph.nodes[node2]['node_type'] == "P"):
                        graph.edges[node1, node2]['data'] = default_p_b_data()
                    elif (graph.nodes[node1]['node_type'] == "B" and graph.nodes[node2]['node_type'] == "R") or (graph.nodes[node1]['node_type'] == "R" and graph.nodes[node2]['node_type'] == "B"):
                        graph.edges[node1, node2]['data'] = default_b_r_data()
                else:
                    messagebox.showerror("Error", "One or both nodes not found")

        plot()
        new_window.destroy() # Close the new window

    # Create the "Submit" button
    submit_button = tk.Button(new_window, text="Submit", command=submit)
    submit_button.pack()

def export_graph():
    file_name = simpledialog.askstring("Input", "Name of File to save (without extension, saves as .gexf):")
    nx.write_gexf(graph, f"{file_name}.gexf")

def import_graph():
    global graph
    file_name = simpledialog.askstring("Input", "Full location of file to import:")
    graph = nx.read_gexf(file_name)
    plot()

def edit_node_attributes():
    node_id = simpledialog.askstring("Input", "Which node do you want to edit?")
    if node_id not in graph:
        messagebox.showerror("Error", "Node not found")
        return
    
    node_attributes = graph.nodes[node_id]['data']
    new_window = tk.Toplevel(window)
    new_window.title("New Window")

    # Create a Text widget for multiline input
    text_widget = tk.Text(new_window, height=5, width=30)
    text_widget.insert("1.0", node_attributes) # Insert default text at the beginning
    text_widget.pack(pady=20)

    # Function to handle the "Submit" button click
    def submit():
        user_input = text_widget.get("1.0", tk.END) # Get all text from the widget
        graph.nodes[node_id]['data'] = user_input
        new_window.destroy() # Close the new window

    # Create the "Submit" button
    submit_button = tk.Button(new_window, text="Submit", command=submit)
    submit_button.pack()

def edit_edge_attributes():
    node1 = simpledialog.askstring("Input", "Enter first node name:")
    node2 = simpledialog.askstring("Input", "Enter second node name:")
    if node1 not in graph or node2 not in graph:
        messagebox.showerror("Error", "Node not found")
        return
    data = graph.edges[node1, node2]['data']
    """Opens a new window with an editable text field."""
    new_window = tk.Toplevel(window)
    new_window.title("New Window")

    # Default text for the text field
    default_text = data

    # Create a Text widget for multiline input
    text_widget = tk.Text(new_window, height=5, width=30)
    text_widget.insert("1.0", default_text) # Insert default text at the beginning
    text_widget.pack(pady=20)

    # Function to handle the "Submit" button click
    def submit():
        user_input = text_widget.get("1.0", tk.END) # Get all text from the widget
        graph.edges[node1, node2]['data'] = user_input
        new_window.destroy() # Close the new window

    # Create the "Submit" button
    submit_button = tk.Button(new_window, text="Submit", command=submit)
    submit_button.pack()

window = Tk() 
window.title('GUI Tool For Creating Graph Topography') 
# dimensions of the main window 
window.geometry("800x500") 

# button that displays the plot 
plot_button = create_button(plot, "Update Plot")
add_node_button = create_button(add_node, "Add Single Node")
mass_create_nodes_button = create_button(mass_create_nodes, "Add Multiple Nodes")
remove_node_button = create_button(remove_node, "Remove Node")
add_edge_button = create_button(add_edge, "Add Edge")
mass_create_edges_button = create_button(mass_create_edges, "Add Multiple Edges")
remove_edge_button = create_button(remove_edge, "Remove Edge")
updateAttributesOfNodes = create_button(edit_node_attributes, "Edit Node Attributes")
updateAttributesOfEdges = create_button(edit_edge_attributes, "Edit Edge Attributes")
export_graph_button = create_button(export_graph, "Export Graph")
import_graph_button = create_button(import_graph, "Import Graph")

plot_button.pack() 
add_node_button.pack()
mass_create_nodes_button.pack()
remove_node_button.pack()
add_edge_button.pack()
mass_create_edges_button.pack()
remove_edge_button.pack()
updateAttributesOfNodes.pack()
updateAttributesOfEdges.pack()
export_graph_button.pack()
import_graph_button.pack()

window.mainloop() 