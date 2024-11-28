import sympy as sp

# Define symbols
theta1, theta2 = sp.symbols('theta1 theta2')
dtheta1, dtheta2 = sp.symbols('dtheta1 dtheta2')
ddtheta1, ddtheta2 = sp.symbols('ddtheta1 ddtheta2')
u = sp.symbols('u')
m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2 = sp.symbols('m1 m2 l1 l2 r1 r2 I1 I2 g f1 f2')

# Define state variables
x1 = sp.Matrix([theta1, theta2])        # Positions
x2 = sp.Matrix([dtheta1, dtheta2])     # Velocities

# Define matrices
M = sp.Matrix([
    [I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) + 2*m2*l1*r2*sp.cos(theta2), I2 + m2*r2**2 + m2*l1*r2*sp.cos(theta2)],
    [I2 + m2*r2**2 + m2*l1*r2*sp.cos(theta2), I2 + m2*r2**2]
])

C = sp.Matrix([
    [-m2*l1*r2*dtheta2*sp.sin(theta2)*(dtheta2 + 2*dtheta1)],
    [m2*l1*r2*sp.sin(theta2)*dtheta1**2]
])

F = sp.Matrix([
    [f1, 0],
    [0, f2]
])

G = sp.Matrix([
    [g*(m1*r1 + m2*l1)*sp.sin(theta1) + g*m2*r2*sp.sin(theta1 + theta2)],
    [g*m2*r2*sp.sin(theta1 + theta2)]
])

U = sp.Matrix([u, 0])  # Input torque vector

# Compute accelerations from dynamics: M * ddtheta + C + F*dtheta + G = U
ddtheta = M.inv() * (U - C - F @ x2 - G)

# Verify the state-space representation
A_upper = -M.inv() @ F
A_lower = sp.eye(2)
A = sp.BlockMatrix([[A_upper, sp.zeros(2, 2)], [A_lower, sp.zeros(2, 2)]]).as_explicit()

B_upper = M.inv()
B = sp.BlockMatrix([[B_upper, sp.zeros(2, 2)], [sp.zeros(2, 2), sp.zeros(2, 2)]]).as_explicit()

# Substitute into state-space dynamics
dx1 = x2
dx2 = ddtheta
dx = sp.Matrix.vstack(dx1, dx2)

state_space_rhs = A @ sp.Matrix.vstack(x1, x2) + B @ U

# Simplification and checks
print("Dynamics derived from M, C, F, G:")
print(dx)
print("\nState-Space Representation:")
print(state_space_rhs)

# Check equivalence
check_equivalence = sp.simplify(dx - state_space_rhs)
print("\nDifference between derived and state-space equations (should be zero):")
print(check_equivalence)
