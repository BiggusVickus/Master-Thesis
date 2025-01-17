from MathFunctions import create_unique_id
class Microorganism:
    def __init__(self, variables):
        self.variables = variables
        self.id = create_unique_id()
        self.population_levels = [variables['population']]
        self.population_growth_rate = variables['growth_rate']
        self.population_death_rate = variables['death_rate']
        self.resource_consumption_rate = variables['resource_consumption_rate']
        self.reproduction_rate = variables['reproduction_rate']
        self.death_rate = variables['death_rate']

    def consume_reosurces(self):
        return self.resource_consumption_rate * self.population_levels[-1]

    def population_reproduce_death(self):
        increase_population = self.death_rate * self.population_levels[-1]
        decrease_population = self.reproduction_rate * self.population_levels[-1]
        new_population_level = increase_population - decrease_population
        self.population_levels.append(new_population_level)
        return new_population_level
    