from Classes.GraphMaker import GraphMaker 

class GraphMakerNonGUI(GraphMaker):
    def __init__(self, seed=None):
        super().__init__(True, seed)
        self.setup_E_S_nodes()
    
    def setup_E_S_nodes(self):
        super().add_node_to_graph("E", "E", self.default_environment_data())
        super().add_node_to_graph("S", "S", self.default_settings_data())
    
    
    def add_node(self, node_type, node_name, node_data = None):
        super().add_node_to_graph(node_type, node_name, node_data)
    
    def remove_node(self, node_name=None):
        return super().remove_node(node_name)
    
    def add_edge(self, node1, node2, edge_data = None):
        return super().add_edge(node1, node2, edge_data)
    
    def remove_edge(self, node1, node2):
        return super().remove_edge(node1, node2)
    
    def mass_create_nodes(self, P=0, B=0, R=0):
        return super().mass_create_nodes(P, B, R)
    
    def mass_create_edges(self, edges_tuple_list, edge_data=None):
        return super().mass_create_edges(edges_tuple_list, edge_data)

    def export_graph_to_file(self, file_name="graph.gexf"):
        return super().export_graph_to_file(file_name)
    
    def import_graph_from_file(self, file_name="graph.gexf"):
        return super().import_graph_from_file(file_name)
    
    def edit_node_attributes(self, node_name_1, node_name_2, node_data):
        return super().edit_node_attributes(node_name_1, node_name_2, node_data)
    
    def edit_edge_attributes(self, node1, node2, user_input):
        return super().edit_edge_attributes(node1, node2, user_input)
    
    def get_graph_object(self):
        return super().get_graph_object()
    
    def set_graph_object(self, graph):
        return super().set_graph_object(graph)