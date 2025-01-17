class Variables:
    def __init__(self, file_location):
        self.txt_location = file_location
    
    def read_txt(self, path):
        with open(path, 'r') as f:
            return f.read()
        
    def determine_type_of_variable(self, string):
        if string == "True":
          return True
        if string == "False":
            return False
        if "[" in string:
            list_maker = []
            string = string.replace('[', '').replace(']', '').replace(' ', '')
            string = string.split(',')
            for element in string:
                list_maker.append(self.determine_type_of_variable(element))
            return list_maker
        if "." in string:
            return float(string)
        if string.isnumeric():
            return int(string)
        else:
            return string
    
    def variables_maker(self, text):
        variable_dictionary = {}
        for row in text.split('\n'):
            if len(row) <= 1:
                continue
            variable = row.split('=')
            variable_name, data = variable[0].strip(), variable[1].strip()
            data_type = self.determine_type_of_variable(data)
            variable_dictionary[variable_name] = data_type

        self.variables = variable_dictionary
        return variable_dictionary

    def load_variables(self):
        variables = self.read_txt(self.txt_location)
        dictionary = self.variables_maker(variables)
        return dictionary

    def save_variables(self, location, dictionary):
        with open(location, 'w') as f:
            for key, value in dictionary.items():
                f.write(key + " = " + str(value) + "\n")