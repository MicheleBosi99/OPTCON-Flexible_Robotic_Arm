import sympy as sp

# Definizione simbolica delle variabili
M1, M2, L1, L2, R1, R2, I1, I2, G, F1, F2 = sp.symbols('M1 M2 L1 L2 R1 R2 I1 I2 G F1 F2')
tau1, tau2 = sp.symbols('tau1 tau2')
theta1, theta2, dtheta1, dtheta2 = sp.symbols('theta1 theta2 dtheta1 dtheta2')

def compute_inertia_matrix(theta2):
    """Calcola la matrice di inerzia M simbolicamente."""
    cos_theta2 = sp.cos(theta2)
    m11 = I1 + I2 + M1 * R1**2 + M2 * (L1**2 + R2**2) + 2 * M2 * L1 * R2 * cos_theta2
    m12 = I2 + M2 * R2**2 + M2 * L1 * R2 * cos_theta2
    m21 = m12
    m22 = I2 + M2 * R2**2
    return sp.Matrix([[m11, m12], [m21, m22]])

def compute_coriolis(theta2, dtheta1, dtheta2):
    """Compute the Coriolis and centrifugal forces matrix C."""
    sin_theta2 = sp.sin(theta2)
    c1 = -M2 * L1 * R2 * dtheta2 * sin_theta2 * (dtheta2 + 2 * dtheta1)
    c2 = M2 * L1 * R2 * sin_theta2 * dtheta1**2
    return sp.Matrix([[c1], [c2]])

def compute_gravity(theta1, theta2):
    """Compute the gravity forces matrix G."""
    sin_theta1 = sp.sin(theta1)
    sin_theta1_theta2 = sp.sin(theta1 + theta2)
    g1 = G * (M1 * R1 + M2 * L1) * sin_theta1 + G * M2 * R2 * sin_theta1_theta2
    g2 = G * M2 * R2 * sin_theta1_theta2
    return sp.Matrix([[g1], [g2]])

# Calcola e stampa la matrice simbolica M
M = compute_inertia_matrix(theta2)
C = compute_coriolis(theta2, dtheta1, dtheta2)
G = compute_gravity(theta1, theta2)

F = sp.Matrix([[F1, 0],
              [ 0 ,F2]])

# Calcola l'inversa simbolica della matrice M
M_inv = M.inv()

Z_2x2 = sp.zeros(2,2)

A = sp.BlockMatrix([[-M_inv @ F, Z_2x2],
                    [sp.eye(2), Z_2x2]])

A = A.as_explicit()

M_inv_ext = sp.BlockMatrix([[M_inv, Z_2x2],
                            [Z_2x2, Z_2x2]])

M_inv_ext = M_inv_ext.as_explicit()

B = M_inv_ext

C_ext = sp.BlockMatrix([[C],
                        [sp.zeros(2, 1)]])

C_ext = C_ext.as_explicit()

G_ext = sp.BlockMatrix([[G],
                        [sp.zeros(2, 1)]])

G_ext = G_ext.as_explicit()

x = sp.Matrix([[dtheta1], [dtheta2], [theta1], [theta2]])
u = sp.Matrix([[tau1],[tau2],[0],[0]])
x_dot = A*x + B*u - M_inv_ext*(C_ext + G_ext)
x_dot = sp.simplify(x_dot)

# Visualizza la matrice di inerzia e la sua inversa simbolicamente
print("")
print("")
print("")
print("Matrice di inerzia simbolica M:")
sp.pprint(M)
print("")
print("")
print("")
print("\nMatrice simbolica di M:")
sp.pprint(M_inv)
print("")
print("")
print("")
print("Matrice simbolica C:")
sp.pprint(C)
print("")
print("")
print("")
print("\nMatrice simbolica di G:")
sp.pprint(G)
print("")
print("")
print("")
print("\nMatrice simbolica di F:")
sp.pprint(F)
print("")
print("")
print("")
print("\nMatrice simbolica di A:")
sp.pprint(A)
print("")
print("")
print("")
print("\nMatrice simbolica di B:")
sp.pprint(B)
print("")
print("")
print("")
print("\nMatrice simbolica di C_ext:")
sp.pprint(C_ext)
print("")
print("")
print("")
print("\nMatrice simbolica di G_ext:")
sp.pprint(G_ext)
print("")
print("")
print("")
print("\nMatrice simbolica di x_dot:")
sp.pprint(x_dot)

