from Classes.Microorganism import Microorganism

class Phage(Microorganism):
    def __init__(self, parameters):
        super().__init__
        self.type = "Phage"
        self.phase_lysic_percentage = 0
        self.targets_bacteria = [] 

    def add_target(self, target_id):
        self.targets_bacteria.append(target_id)

    def remove_target(self, target_id):
        self.targets_bacteria.pop(target_id)