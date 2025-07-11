import pickle
from collections import OrderedDict
import h5py
import operator
import io
import pandas as pd

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
        memory_file = io.BytesIO()
        with self.hdf_data as hf_old:
            with h5py.File(memory_file, 'w') as hf_new:
                hf_new.attrs.update(hf_old.attrs)
                for datagroup_name, datagroup in hf_old.items():
                    result_group = hf_new.create_group(datagroup_name)
                    # Copy attributes
                    for attr_name, attr_value in datagroup.attrs.items():
                        result_group.attrs[attr_name] = attr_value
                    # Copy datasets
                    for dataset_name, dataset_data in datagroup.items():
                        result_group.create_dataset(dataset_name, data=dataset_data[()])
            memory_file.seek(0)
        return memory_file
    
    def create_filter(self, datagroup, parameter_names, comparisons, value_searches):
        conditions = [
            self.evaluate_operator(comp)(datagroup.attrs[param], val) 
            for param, comp, val in zip(parameter_names, comparisons, value_searches)
        ]
        return conditions

    def simple_query(self, dataset, parameter_name, comparison, value_search):
        memory_file = io.BytesIO()
        op = self.evaluate_operator(comparison)
        with h5py.File(dataset, 'r') as hf_old:  
            with h5py.File(memory_file, 'w') as hf_new:
                hf_new.attrs.update(hf_old.attrs)
                for datagroup_name, datagroup in hf_old.items():
                    if op(datagroup.attrs[parameter_name], value_search):
                        result_group = hf_new.create_group(datagroup_name)
                        result_group.attrs.update(datagroup.attrs)
                        for dataset_name, dataset_data in datagroup.items():
                            result_group.create_dataset(dataset_name, data=dataset_data[()])
        memory_file.seek(0)
        return memory_file
    
    def and_query(self, dataset, parameter_names, comparisons, value_searches):
        memory_file = io.BytesIO()
        with h5py.File(dataset, 'r') as hf_old:  
            with h5py.File(memory_file, 'w') as hf_new:
                hf_new.attrs.update(hf_old.attrs)
                for datagroup_name, datagroup in hf_old.items():
                    if all(self.create_filter(datagroup, parameter_names, comparisons, value_searches)):
                        result_group = hf_new.create_group(datagroup_name)
                        result_group.attrs.update(datagroup.attrs)
                        for dataset_name, dataset_data in datagroup.items():
                            result_group.create_dataset(dataset_name, data=dataset_data[()])
        memory_file.seek(0)
        return memory_file

    def or_query(self, dataset, parameter_names, comparisons, value_searches):
        memory_file = io.BytesIO()
        with h5py.File(dataset, 'r') as hf_old:  
            with h5py.File(memory_file, 'w') as hf_new:
                hf_new.attrs.update(hf_old.attrs)
                for datagroup_name, datagroup in hf_old.items():
                    if any(self.create_filter(datagroup, parameter_names, comparisons, value_searches)):
                        result_group = hf_new.create_group(datagroup_name)
                        result_group.attrs.update(datagroup.attrs)
                        for dataset_name, dataset_data in datagroup.items():
                            result_group.create_dataset(dataset_name, data=dataset_data[()])
        memory_file.seek(0)
        return memory_file

    def finalize_query(self, dataset, format='dictionary'):
        if format == 'dictionary':
            dictionary = {}
            with h5py.File(dataset, 'r') as data:
                for datagroup_name, datagroup in data.items():
                    dictionary[datagroup_name] = {}
                    # for attr_name, attr_value in datagroup.attrs.items():
                    dictionary[datagroup_name]['parameters'] = {}
                    for attr_name, attr_value in datagroup.attrs.items():
                        dictionary[datagroup_name]['parameters'][attr_name] = attr_value
                    for dataset_name, dataset_data in datagroup.items():
                        dictionary[datagroup_name][dataset_name] = dataset_data[()]
            return dictionary
        elif format == 'dataframe':
            row_list = []
            with h5py.File(dataset, 'r') as data:
                for datagroup_name, datagroup in data.items():
                    dict1 = OrderedDict()
                    dict1['results'] = datagroup_name
                    for attr_name, attr_value in datagroup.attrs.items():
                        dict1[attr_name] = attr_value
                    dict1['t_values'] = datagroup['t_values'][()]
                    dict1['y_values'] = datagroup['y_values'][()]
                    row_list.append(dict1)
            df = pd.DataFrame(row_list)
            return df

    def export_query_as_hdf(self, data, filename='query.hdf5'):
        if isinstance(data, io.BytesIO):
            data.seek(0)
        if not filename.endswith('.hdf5'):
            filename = filename + '.hdf5'
        with h5py.File(data, 'r') as source:
            with h5py.File(filename, 'w') as target:
                for attr_name, attr_value in source.attrs.items():
                    target.attrs[attr_name] = attr_value
                for group_name, group in source.items():
                    target_group = target.create_group(group_name)
                    for attr_name, attr_value in group.attrs.items():
                        target_group.attrs[attr_name] = attr_value
                    for dataset_name, dataset_data in group.items():
                        target_group.create_dataset(dataset_name, data=dataset_data[()])
