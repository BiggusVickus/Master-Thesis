import numpy as np
def optical_density(array, list_of_names):
    optical_list = []
    counter = 0
    for j, item in enumerate(array):
        if list_of_names[counter].lower() in ["uninfected", "infected", "bacteria", "b", "u", "i", "b0", "u0", "i0", "infect", "uninf", "inf", "uninfect", "uninfected bacteria", "infected bacteria", "bacteria uninfected", "bacteria infected", "bacteria uninf", "bacteria infect", "bacteria u", "bacteria i", "bacteria u0", "bacteria i0", "bacteria infect", "bacteria uninfected"]:
            optical_list.append(item)
        counter += 1
    initial_sum = optical_list[0][0]
    for i in range(len(optical_list)):
        for j in range(len(optical_list[i])):
            if i == 0 and j == 0:
                continue
            initial_sum += optical_list[i][j]
    return initial_sum
# Define models
def log_func(self, x, a, c):
    return a * np.log(x) + c

def lin_func(x, a, c):
    return a * x + c

def serial_transfer_calculation(graph_data, original_final_simulation_output, serial_transfer_value, serial_tranfer_option, flattened):
        row_of_names = []
        row_of_values = []
        for key, value in graph_data.items():
            row_of_names += [key] * value["data"].size
        if (len(serial_tranfer_option) > 0):
            return flattened + original_final_simulation_output / serial_transfer_value
        for final, name, flat in zip(original_final_simulation_output, row_of_names, flattened):
            if (name.lower() in ["resources", "resource", "r", "res", "r0", "nutrient", "nutrients", "n", "nut", "n0"]):
                row_of_values.append(flat + final / serial_transfer_value)
            else:
                row_of_values.append(final / serial_transfer_value)
        return row_of_values
    
def sum_up_columns(unflattened_data, value_add_column):
    if value_add_column not in [None, False]:
        unflattened_temp = []
        for i in range(0, len(unflattened_data), value_add_column):
            unflattened_temp.append(np.sum(unflattened_data[i:i + value_add_column], axis=0))
        return unflattened_temp
    else:
        return unflattened_data
    
def split_comma_minus(input, range, steps, use_opt_1_or_opt_2):
    if use_opt_1_or_opt_2:
        return [float(value.strip()) for value in input.split(",")]
    else:
        start_1, end_1 = [float(value.strip()) for value in range.split("-")]
        return np.linspace(start_1, end_1, int(steps)).tolist()
    
def unifrom_color_gradient_maker(i, n):
    ratio = i / (n - 1) if n > 1 else 0  # avoid division by zero
    r = int(255 * (1 - ratio))  # interpolate from red
    g = int(255 * ratio)        # interpolate to green
    b = 0                       # no blue component
    return f'rgb({r},{g},{b})'

def determine_type_of_variable(string):
    string = string.strip()
    if string == "True":
        return True
    if string == "False":
        return False
    if string[0] == "[":
        list_maker = []
        string = string.replace('[', '').replace(']', '').replace(' ', '')
        string = string.split(',')
        for element in string:
            list_maker.append(determine_type_of_variable(element))
        return list_maker
    if "." in string:
        return float(string)
    if string.isnumeric():
        return int(string)
    else:
        return string