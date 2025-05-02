from joblib import Parallel, delayed

class ParallelComputing:
    """
    This class is used to perform parallel computing tasks.
    """

    def __init__(self, filename_location=None):
        self.filename_location = filename_location

    
    def run_parallel(self, iter_items, unique_param_names, graph_data, vector_items_of_name, matrix_items_of_names, initial_condition, analysis, other_parameters_to_pass, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
        results = Parallel(n_jobs=-1)(delayed(self.process_combinations)(x, unique_param_names, graph_data, vector_items_of_name, matrix_items_of_names, initial_condition, analysis, other_parameters_to_pass, non_graphing_data_vectors, non_graphing_data_matrices, environment_data) for x in iter_items)
        results_t, results_y = zip(*results)
        return results_t, results_y
    
    def process_combinations(self, param_combination, unique_param_names, graph_data, vector_items_of_name, matrix_items_of_names, initial_condition, analysis, other_parameters_to_pass, non_graphing_data_vectors, non_graphing_data_matrices, environment_data):
        # print(f"Processing combination: {param_combination}")
        items_of_name = []
        for key, value in graph_data.items():
            items_of_name += [key] * value["data"].size
        for param_name, param_value in zip(unique_param_names, param_combination):
            if param_name in items_of_name:
                indexes = [i for i, item in enumerate(items_of_name) if item == param_name]
                for index in indexes:
                    initial_condition[index] = param_value
            elif param_name in vector_items_of_name:
                non_graphing_data_vectors[vector_items_of_name.index(param_name)][:] = param_value
            elif param_name in matrix_items_of_names:
                non_graphing_data_matrices[matrix_items_of_names.index(param_name)][:][:] = param_value
            elif param_name in environment_data:
                environment_data[param_name] = param_value
        solved_system = analysis.solve_system(analysis.odesystem, initial_condition, analysis, *other_parameters_to_pass, *non_graphing_data_vectors, *non_graphing_data_matrices,  environment_data)
        return solved_system.t, solved_system.y