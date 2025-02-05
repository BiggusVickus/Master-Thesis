from Classes.GraphMaker import GraphMaker 

class GraphMakerNonGUI(GraphMaker):
    def __init__(self):
        super().__init__(True)
    
    def add_node(self, node_type, node_name, node_data = None):
        super().add_node(node_type, node_name, node_data)
    
    def remove_node(self, node_name=None):
        return super().remove_node(node_name)
    
    def add_edge(self, node1, node2, edge_data = None):
        return super().add_edge(node1, node2, edge_data)
    
    def remove_edge(self, node1, node2):
        return super().remove_edge(node1, node2)
    
    def mass_create_nodes(self, P=0, B=0, R=0):
        return super().mass_create_nodes(P, B, R)
    
    def mass_create_edges(self, edge_connections):
        return super().mass_create_edges(edge_connections)

    def export_graph_to_file(self, file_name="graph.gexf"):
        return super().export_graph_to_file(file_name)
    
    def import_graph_from_file(self, file_name="graph.gexf"):
        return super().import_graph_from_file(file_name)
    
    def edit_node_attributes(self, node_name_1, node_name_2, node_data):
        return super().edit_node_attributes(node_name_1, node_name_2, node_data)
    
    def edit_edge_attributes(self, node1, node2, user_input):
        return super().edit_edge_attributes(node1, node2, user_input)
    