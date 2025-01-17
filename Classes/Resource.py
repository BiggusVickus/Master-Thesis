from MathFunctions import create_unique_id
import Classes.Bacteria as Bacteria
class Resource:
    def __init__(self, parameters):
        self.id = create_unique_id
        self.concentration = [parameters['starting_concentration']]
        self.added_concentration = parameters['increasing concentration']
        self.reduced_by: Bacteria = []
    
    def add_reduce_concentration(self, add, reduce):
        new_concentration = self.concentration[-1] + add + reduce
        self.concentration.append(new_concentration)
        return new_concentration
    
    def concentration_added(self):
        return self.added_concentration
    
    def add_affected_by_bacteria(self, bacteria_id):
        self.reduced_by.append(bacteria_id)

    def remove_affected_by_bacteria(self, bacteria_id):
        self.reduced_by.pop(bacteria_id)