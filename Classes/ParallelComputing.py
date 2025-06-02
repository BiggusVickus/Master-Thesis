from joblib import Parallel, delayed
import numpy as np
from copy import deepcopy

class ParallelComputing:
    """
    This class is used to perform parallel computing tasks.
    """
    def __init__(self):
        pass
    
    def run_parallel(self, iter_items, unique_param_names, graph_data, initial_condition, vector_data, vector_names, matrix_data, matrix_names, analysis, other_parameters_to_pass, environment_data):
        results = Parallel(n_jobs=1)(delayed(self.process_combinations)(x, unique_param_names, graph_data, deepcopy(initial_condition), deepcopy(vector_data), vector_names, deepcopy(matrix_data), matrix_names, analysis, other_parameters_to_pass, deepcopy(environment_data)) for x in iter_items)
        results_t, results_y = zip(*results)
        return results_t, results_y
    
    def process_combinations(self, param_combination, unique_param_names, graph_data, initial_condition, vector_data, vector_names, matrix_data, matrix_names, analysis, other_parameters_to_pass, environment_data):
        items_of_name = []
        for key, value in graph_data.items():
            items_of_name += [key] * value["data"].size
        for param_name, param_value in zip(unique_param_names, param_combination):
            if param_value == np.inf:
                continue
            if param_name in items_of_name:
                indexes = [i for i, item in enumerate(items_of_name) if item == param_name]
                for index in indexes:
                    initial_condition[index] = param_value
            elif param_name in vector_names:
                vector_data[vector_names.index(param_name)][:] = param_value
            elif param_name in matrix_names:
                matrix_data[matrix_names.index(param_name)][:][:] = param_value
            elif param_name in environment_data:
                environment_data[param_name] = param_value
        solved_system = analysis.solve_system(analysis.odesystem, initial_condition, analysis, *other_parameters_to_pass, *vector_data, *matrix_data, environment_data)
        return solved_system.t, solved_system.y