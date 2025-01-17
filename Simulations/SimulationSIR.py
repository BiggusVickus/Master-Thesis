import Classes.Bacteria as Bacteria
import Classes.Phage as Phage
import Classes.Resource as Resource
class SimulationSIR():
    def _init_(self, dictionary):
        self.bacteria_pop = dictionary
        self.time = dictionary['time']
        self.bacteria_agents: Bacteria = []
        self.phage_agents: Phage = []
        self.resource_agents: Resource = []

    def add_agent(self, agent, type):
        if type == "Bacteria":
            self.bacteria_agents.append(agent)
        if type == "Phage":
            self.phage_agents.append(agent)
        if type == "Resource":
            self.resource_agents.append(agent)
    
    def remove_agent(self, agent, type):
        if type == "Bacteria":
            self.bacteria_agents.pop(agent)
        if type == "Phage":
            self.phage_agents.pop(agent)
        if type == "Resource":
            self.resource_agents.pop(agent)

    def bacteria(self):
        for bacteria in self.bacteria_agents:
            bacteria.population_reproduce_death()
            bacteria.become_infected()

    def phages(self):
        for phage in self.phage_agents:
            phage.population_reproduce_death()

    def resources(self):
        for resource in self.resource_agents:
            for bacteria in self.bacteria_agents: 
                if bacteria.id not in resource.reduced_by:
                    continue
                reduce_resources = bacteria.consume_resources()
                add_resource = resource.added_concentration()
                resource.add_reduce_concentration(add_resource, reduce_resources)

    def run(self):
        for _ in range(self.time):
            self.bacteria()
            self.phages()
            self.resources()
