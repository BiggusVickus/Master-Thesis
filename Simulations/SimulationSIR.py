class SimulationSIR():
    def _init_(self, dictionary):
        self.bacteria_pop = dictionary
        self.time = dictionary['time']
        self.bacteria_agents = []
        self.phage_agents = []
        self.resource_agents = []

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

    def main_loop(self):
        for i in range(self.time):
            