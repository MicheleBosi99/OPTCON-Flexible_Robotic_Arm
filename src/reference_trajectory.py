import numpy as np
import dynamics as dyn

def sigmoid(k):
    """Calculate the sigmoid function."""
    return 1 / (1 + np.exp(-k))

def generate_trajectory(tf, x_goal, k_eq, dt=1e-3):
    """
    Generate a smooth trajectory for each element in x_goal using a sigmoid function,
    and compute control input as the sine of x_reference scaled by k_eq.

    Args:
        tf (float): Total time for the trajectory.
        x_goal (np.ndarray): Target goal states for each dimension.
        k_eq (float): Scaling factor for the control input.
        dt (float): Time step for sampling.

    Returns:
        x_reference (np.ndarray): Trajectory values for each time step.
        u_reference (np.ndarray): Control input values for each time step.
    """
    total_time_steps = int(tf / dt)
    time = np.linspace(0, tf, total_time_steps)
    x_size = x_goal.shape[0]

    # Initialize references
    x_reference = np.zeros((x_size, total_time_steps))
    u_reference = np.zeros((x_size, total_time_steps))

    for i in range(x_size):
        # Sigmoid function for each dimension scaled to [0, x_goal[i]]
        x_reference[i, :] = x_goal[i] * sigmoid(10 * (time / tf - 0.5))
        # Control input as sine of the state scaled by k_eq
        u_reference[i, :] = k_eq * np.sin(x_reference[i, :])

    return x_reference, u_reference
