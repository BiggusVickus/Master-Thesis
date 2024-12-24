class Variables:
    def __init__(self, file_location):
        self.txt_location = file_location
    
    def read_txt(self, path):
        with open(path, 'r') as f:
            return f.read()
    
    def variables_maker(self, text):
        variable_dictionary = {}
        for row in text.split('\n'):
            if len(row) <= 1:
                continue
            variable = row.split('=')
            variable_name, data = variable[0].strip(), variable[1].strip()
            if data.isnumeric():
                variable_dictionary[variable_name] = int(data)
            else:
                variable_dictionary[variable_name] = data
        self.variables = variable_dictionary
        return variable_dictionary

    def load_variables(self):
        txt_location = self.txt_location
        variables = self.read_txt(txt_location)
        dictionary = self.variables_maker(variables)
        return dictionary

    def save_variables(self, location, dictionary):
        with open(location, 'w') as f:
            for key, value in dictionary.items():
                print(key, value)
                f.write(key + " = " + str(value) + "\n")