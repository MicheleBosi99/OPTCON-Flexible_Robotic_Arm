from dynamics import dynamics as dyn
from visualizer import animation as anim
import numpy as np

# Pendulum lengths
L1 = 1.5
L2 = 1.5

# Simulation parameters
FRAME_SKIP = 20
t_i = 0
t_f = 10
dt = 1e-4

def main():
    """
    Main function to simulate the dynamics and visualize the results.
    """
    # Initial state and input
    x_0 = np.array([[3.14], [0], [0], [0]])  # Initial state
    u_0 = np.array([[10], [0], [0], [0]])   # Constant input

    time_intervals = int((t_f - t_i) / dt + 1)

    x_history = [x_0]

    # Compute dynamics for each time step
    print("Computing dynamics...")
    for i in range(1, time_intervals):
        x_history.append(dyn(x_history[i - 1], u_0, dt))

    # Convert state history to a matrix
    matrix_x_history = np.hstack(x_history)

    # Visualize the simulation
    anim(matrix_x_history.T, L1, L2, FRAME_SKIP)

if __name__ == "__main__":
    main()
