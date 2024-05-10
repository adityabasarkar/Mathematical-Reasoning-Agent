import sympy as sp

# Define the variable for the side length
s = sp.symbols('s', real=True, positive=True)

# Define the equations based on the distances from P to the vertices A, B, C, and E
# PA = sqrt(70), PB = sqrt(97), PC = sqrt(88), PE = sqrt(43)
# Coordinates of P are (x, y, z)
# A = (0,0,0), B = (s,0,0), C = (s,s,0), E = (0,0,s)
# PA^2 = x^2 + y^2 + z^2
# PB^2 = (x-s)^2 + y^2 + z^2
# PC^2 = (x-s)^2 + (y-s)^2 + z^2
# PE^2 = x^2 + y^2 + (z-s)^2

# Define the variables for the coordinates of P
x, y, z = sp.symbols('x y z', real=True)

# Define the equations
eq1 = sp.Eq(x**2 + y**2 + z**2, 70)
eq2 = sp.Eq((x-s)**2 + y**2 + z**2, 97)
eq3 = sp.Eq((x-s)**2 + (y-s)**2 + z**2, 88)
eq4 = sp.Eq(x**2 + y**2 + (z-s)**2, 43)

# Solve the system of equations
solutions = sp.solve((eq1, eq2, eq3, eq4), (x, y, z, s))

# Print the solutions for s
valid_solutions = [sol[s] for sol in solutions if sol[s] > 0]
print("Valid solutions for s:", valid_solutions)