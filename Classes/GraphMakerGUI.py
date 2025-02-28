from tkinter import * 
import tkinter as tk
from tkinter import simpledialog, messagebox
import networkx as nx
import numpy as np
from Classes.GraphMaker import GraphMaker 
np.random.seed(69)

class GraphMakerGUI(GraphMaker):
    def __init__(self):
        super().__init__(True)
        self.window = Tk() 
        self.window.geometry("1000x1000")
        self.window.wm_title("GUI Tool For Creating Network Topography")
        self.setup_E_node()
        self.initialize_GUI()
    
    def setup_E_node(self):
        super().add_node_to_graph("E", "E", self.default_environment_data())
    
    def initialize_GUI(self):
        self.window.title('GUI Tool For Creating Network Topography') 
        # dimensions of the main window 
        self.window.geometry("800x500") 

        # button that displays the plot 
        plot_button = self.create_button(self.plot, "Update Plot")
        add_node_button = self.create_button(self.add_node, "Add Single Node")
        mass_create_nodes_button = self.create_button(self.mass_create_nodes, "Add Multiple Nodes")
        remove_node_button = self.create_button(self.remove_node, "Remove Node")
        add_edge_button = self.create_button(self.add_edge, "Add Edge")
        mass_create_edges_button = self.create_button(self.mass_create_edges, "Add Multiple Edges")
        remove_edge_button = self.create_button(self.remove_edge, "Remove Edge")
        updateAttributesOfNodes = self.create_button(self.edit_node_attributes, "Edit Node Attributes")
        updateAttributesOfEdges = self.create_button(self.edit_edge_attributes, "Edit Edge Attributes")
        export_graph_button = self.create_button(self.export_graph_to_file, "Export Graph")
        import_graph_button = self.create_button(self.import_graph_from_file, "Import Graph")

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
        self.plot()
        self.window.mainloop() 

    def create_button(self, command, text, row = 0, column = 0):
        button = Button(master = self.window,
                        command = command, 
                        height = 1,
                        width = 10, 
                        text = text) 
        return button
    
    def add_node(self):
        node_type = simpledialog.askstring("Add Node: Node Type", "Enter exactly P, B, R for type of node type to input:")
        node_name = simpledialog.askstring("Add Node: Node Name", "Enter node name:")
        super().add_node_to_graph(node_type, node_name)
        self.plot()

    def remove_node(self):
        node_name = simpledialog.askstring("Remove Node", "Enter node name to remove:")
        super().remove_node(node_name)
        self.plot()

    def add_edge(self, node1=None, node2=None, edge_data=None):
        if node1 is None:
            node1 = simpledialog.askstring("Add Edge", "Enter node 1:")
        if node2 is None:
            node2 = simpledialog.askstring("Add Edge", "Enter node 2:")
        super().add_edge(node1, node2, edge_data)
        self.plot()

    def remove_edge(self):
        node1 = simpledialog.askstring("Remove Edge", "Enter node 1:")
        node2 = simpledialog.askstring("Remove Edge", "Enter node 2:")
        super().remove_edge(node1, node2)
        self.plot()
    
    def mass_create_nodes(self):
        P = simpledialog.askinteger("Add Phage Nodes", "Enter number of P nodes:")
        B = simpledialog.askinteger("Add Bacteria Nodes", "Enter number of B nodes:")
        R = simpledialog.askinteger("Add Resource Nodes", "Enter number of R nodes:")
        super().mass_create_nodes(P, B, R)
        self.plot()
    
    def mass_create_edges(self):
        def submit():
            error_text = ""
            submitted_text = text_widget.get("1.0", tk.END) # Get all text from the widget
            new_window.destroy() # Close the new window
            edge_tuples = []
            lines = submitted_text.split("\n")
            for line in lines:
                if line == "":
                    continue
                edge = line.split(" ")
                if len(edge) < 2:
                    error_text += "There arent at least 2 lines in line: " + line + ", skipping\n"
                    continue
                for i in range(1, len(edge)):
                    if edge[i].strip() == "" :
                        continue
                    edge_tuples.append((edge[0], edge[i]))
            error_text = GraphMaker.mass_create_edges(self, edge_tuples)
            self.plot()
            if error_text != "":
                error_text = "Found issues with these edges, others have been successfully added: \n\n" + error_text
                return self.error_message(error_text, message_type="Warning")
            return

        new_window = tk.Toplevel(self.window)
        new_window.title("Create Multiple Edges")
        text_widget = tk.Text(new_window, height=10, width=50)
        text_widget.pack(pady=20)

        submit_button = tk.Button(new_window, text="Submit", command=submit)
        submit_button.pack()
    
    def export_graph_to_file(self):
        file_name = simpledialog.askstring("Save File to Disk", "Name of File to save (accepted = .gexf, .gml, .graphml, .net):")
        return super().export_graph_to_file(file_name)
    
    def import_graph_from_file(self):
        file_name = simpledialog.askstring("Import Grpah File", "Full location of file to import:")
        return super().import_graph_from_file(file_name)
    
    def edit_node_attributes(self):
        node_name = simpledialog.askstring("Edit Node Attribute", "Which node do you want to edit?").strip()
        def submit():
            submitted_text = text_widget.get("1.0", tk.END) # Get all text from the widget
            new_window.destroy() # Close the new window
            GraphMaker.edit_node_attributes(self, node_name, submitted_text)
            return submitted_text
        if (not self.graph.has_node(node_name)):
            return messagebox.showerror("Error", "Node does not exist.")
        node_attributes = self.graph.nodes[node_name]['data']

        new_window = tk.Toplevel(self.window)
        new_window.title("Edit Node Attribute of " + node_name)
        text_widget = tk.Text(new_window, height=10, width=50)
        text_widget.insert("1.0", node_attributes)
        text_widget.pack(pady=40)

        submit_button = tk.Button(new_window, text="Submit", command=submit)
        submit_button.pack()

    def edit_edge_attributes(self):
        node1 = simpledialog.askstring("Edit Edge Attribute", "Enter first node name:")
        node2 = simpledialog.askstring("Edit Edge Attribute", "Enter second node name:")
        if node1 == "" or node2 == "" or node1 is None or node2 is None:
            return messagebox.showerror("Error", "Please enter both node names.")
        node1 = node1.strip()
        node2 = node2.strip()
        def submit():
            submitted_text = text_widget.get("1.0", tk.END) # Get all text from the widget
            new_window.destroy() # Close the new window
            GraphMaker.edit_edge_attributes(self, node1, node2, submitted_text)
            return submitted_text
        
        if not self.graph.has_edge(node1, node2):
            return messagebox.showerror("Error", "Edge does not exist.")
        edge_attributes = self.graph.edges[node1, node2]['data']

        new_window = tk.Toplevel(self.window)
        new_window.title("Edit Edge Atrribute of " + node1 + " and " + node2)
        text_widget = tk.Text(new_window, height=10, width=50)
        text_widget.insert("1.0", edge_attributes)
        text_widget.pack(pady=20)

        submit_button = tk.Button(new_window, text="Submit", command=submit)
        submit_button.pack()
    
    def get_graph_object(self):
        return super().get_graph_object()

    def set_graph_object(self, graph):
        return super().set_graph_object(graph)

    def export_graph(self, file_name):
        return super().export_graph_to_file(file_name)
    
    def import_graph(self, file_name):
        return super().import_graph_from_file(file_name)