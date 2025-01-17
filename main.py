from Classes.Variables import *

location = "variables.txt"
vars = Variables(location)
varss = vars.load_variables()
print(varss)