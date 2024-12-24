class Microorganism:
    def __init__(self, variables):
        self.variables = variables
        self.population_counts = variables['population']
        self.population_growth = variables['growth_rate']
        self.population_death = variables['death_rate']
        self.population_levels = [self.population_counts]
        self.resource_consumption_rate = variables['resource_consumption_rate']
        self.reproduction_rate = variables['reproduction_rate']
        self.death_rate = variables['death_rate']

    def consume_reosurce(self):
        return self.resource_consumption_rate

    def reproduce(self):
        return self.reproduction_rate

    def die(self):
        return self.death_rate
    
