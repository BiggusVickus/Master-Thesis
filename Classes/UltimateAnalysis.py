import pickle
import numpy as np
import h5py
import operator
import io

class UltimateAnalysis:
    def __init__(self):
        self.pickle_file_location = None
        self.dictionary = None
        self.parameter_names_used = None
        self.analysis = None
        self.graph_data = None
        self.non_graph_data_vector = None
        self.non_graph_data_matrix = None
        self.settings = None
        self.environment_data = None
        self.other_parameters_to_pass = None
        self.hdf_file_location = None
        self.hdf_data = None
    
    def test_near(value, error):
        if (value - error) <= value <= (value + error):
            return True
        return False


    def evaluate_operator(self, operator_name):
            if (operator_name == '<'):
                return operator.lt
            elif operator_name == '<=':
                return operator.le
            elif operator_name == '>':
                return operator.gt
            elif operator_name == '>=':
                return operator.ge
            elif operator_name == '==':
                return operator.eq
            elif operator_name == '!=':
                return operator.ne
            elif operator_name == '~=':
                return self.test_near
        
    def unpack_pickle(self, pickle_file_location):
        self.pickle_file_location = pickle_file_location
        with open(pickle_file_location, 'rb') as f:
            # Unpickle the data
            self.dictionary = pickle.load(f)
        self.parameter_names_used = self.dictionary['parameter_names_used']
        self.parameter_values_tested = self.dictionary['parameter_values_tested']
        self.analysis = self.dictionary['analysis']
        self.graph_data = self.dictionary['graph_data']
        self.non_graph_data_vector = self.dictionary['non_graph_data_vector']
        self.non_graph_data_matrix = self.dictionary['non_graph_data_matrix']
        self.settings = self.dictionary['settings']
        self.environment_data = self.dictionary['environment_data']
        self.other_parameters_to_pass = self.dictionary['other_parameters']
        self.hdf_file_location = self.dictionary['hdf_file_location']

        self.hdf_data = h5py.File(self.hdf_file_location, 'r')
        return self.dictionary
        
    def new_query(self):
        return self.hdf_data
    
    def simple_query(self, dataset, parameter_name, comparison, value_search, error=0):
        memory_file = io.BytesIO()
        op = self.evaluate_operator(comparison)
        with h5py.File(memory_file, 'w') as hf_new:
            for datagroup_name, datagroup in dataset.items():
                if op(datagroup.attrs[parameter_name], value_search):
                    # print(f"Dataset: {datagroup_name}, Parameter: {parameter_name}, Value: {datagroup.attrs[parameter_name]}")
                    result_group = hf_new.create_group(datagroup_name)
                    # Copy attributes
                    for attr_name, attr_value in datagroup.attrs.items():
                        result_group.attrs[attr_name] = attr_value
                    
                    # Copy datasets
                    for dataset_name, dataset_data in datagroup.items():
                        result_group.create_dataset(dataset_name, data=dataset_data[()])
        memory_file.seek(0)
        return memory_file

    def and_query(self, dataset, parameter_name, comparison, value):
        pass

    def or_query(self, dataset, parameter_name, comparison, value):
        pass

    def finalize_query(self, dataset, ):
        pass

    def export_query(self, data, filename):
        with h5py.File(filename, 'w') as hf:
            hf.write(data.getbuffer())