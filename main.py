from Classes.Variables import *

location = "variables.txt"
vars = Variables(location)
varss = vars.load_variables()
phage_population = []
bacteria_population = []

for i in range(varss['time']):
    