import Classes.Microorganism as Microorganism
import Classes.Phage as Phage
class Bacteria(Microorganism):
    def __init__(self):
        """
            Class representing Bacteria and their functionalities
            Args:
                Microorganism (parameters): input parameters for the Bacteria
        """
        super().__init__
        self.type = "Bacteria"
        self.infected_by_phages: Phage = []
        self.population_level_infected: Bacteria = [1]
    
    def add_affected_by_phage(self, phage_id):
        self.infected_by_phages.append(phage_id)

    def remove_affected_by_phage(self, phage_id):
        self.infected_by_phages.pop(phage_id)
    
    def population_reproduce_death(self):
        increase_population = self.death_rate * self.population_levels[-1]
        decrease_population = self.reproduction_rate * self.population_levels[-1]
        new_population_level = increase_population - decrease_population
        self.population_levels.append(new_population_level)
        return new_population_level
    
    def death_by_phages(self, phages):
        for phage in phages:
            if phage.id in self.infected_by_phages():
                