import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
T = 10
dt = 1e-4
iterations = int(T/dt)

# Dynamics parameters
M1 = 2
M2 = 2
L1 = 1.5
L2 = 1.5
R1 = 0.75
R2 = 0.75
I1 = 1.5
I2 = 1.5
G = 9.81
F1 = 0.1
F2 = 0.1

F = np.array([[F1, 0],
              [0, F2]])

Z_2x2 = np.zeros((2, 2))

def getLengths():
    return L1, L2

def compute_inertia_matrix(theta2):
    """Compute the inertia matrix M."""
    cos_theta2 = np.cos(theta2)
    m11 = I1 + I2 + M1 * R1**2 + M2 * (L1**2 + R2**2) + 2 * M2 * L1 * R2 * cos_theta2
    m12 = I2 + M2 * R2**2 + M2 * L1 * R2 * cos_theta2
    m21 = m12
    m22 = I2 + M2 * R2**2
    return np.array([[m11, m12], [m21, m22]])

def compute_coriolis(theta2, dtheta1, dtheta2):
    """Compute the Coriolis and centrifugal forces matrix C."""
    sin_theta2 = np.sin(theta2)
    c1 = -M2 * L1 * R2 * sin_theta2 * (dtheta2 + 2 * dtheta1) * dtheta2
    c2 = M2 * L1 * R2 * sin_theta2 * dtheta1**2
    return np.array([[c1], [c2]])

def compute_gravity(theta1, theta2):
    """Compute the gravity forces matrix G."""
    sin_theta1 = np.sin(theta1)
    sin_theta1_theta2 = np.sin(theta1 + theta2)
    g1 = G * (M1 * R1 + M2 * L1) * sin_theta1 + G * M2 * R2 * sin_theta1_theta2
    g2 = G * M2 * R2 * sin_theta1_theta2
    return np.array([[g1], [g2]])

def dynamics(x, u, dt=1e-3):

    #Extract Elements
    dtheta1 = x[0].item()
    dtheta2 = x[1].item()
    theta1 = x[2].item()
    theta2 = x[3].item()

    

    # Compute matrices
    M = compute_inertia_matrix(theta2)
    M_inv = np.linalg.inv(M)
    C = compute_coriolis(theta2, dtheta1, dtheta2)
    G = compute_gravity(theta1, theta2)
    
    A = np.block([[ -M_inv @ F, Z_2x2 ], 
                  [ np.eye(2), Z_2x2 ]])
       
    
    M_inv_ext = np.block([
        [M_inv, Z_2x2],
        [Z_2x2, Z_2x2]
    ])

    B = M_inv_ext
    
    C_ext = np.block([
        [C],
        [np.zeros((2, 1))] 
    ])
    
    G_ext = np.block([
        [G],
        [np.zeros((2, 1))]
    ])
    
    x = x.reshape(-1, 1)
    u = u.reshape(-1, 1)

    x_dot = A @ x + B @ u - M_inv_ext @ (C_ext + G_ext)
    x_new = x + dt * x_dot

    #jacobian_x_dot = compute_jacobian(dtheta1, dtheta2, theta1, theta2, u[0].item())
    return np.transpose(x_new)

def natural_evolution(x, u):
    xe = np.zeros([iterations, 4])
    xe[0, :] = x

    for i in range(1, iterations):
        xe[i] = dynamics(xe[i-1, :] , u)

    return xe