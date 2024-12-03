import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animation(x_history, L1, L2,FRAME_SKIP=20):
    """
    Create an animation of a double pendulum.

    Parameters:
        x (numpy.ndarray): Array containing the simulation data with shape (T, 4).
        L1 (float): Length of the first pendulum rod.
        L2 (float): Length of the second pendulum rod.
    """
    T = x_history.shape[0]
    data = np.zeros((T, 4))

    # Calculate positions of the pendulum
    data[:, 0] = L1 * np.sin(x_history[:, 2])  # x1
    data[:, 1] = -L1 * np.cos(x_history[:, 2])  # y1
    data[:, 2] = data[:, 0] + L2 * np.sin(x_history[:, 3])  # x2
    data[:, 3] = data[:, 1] - L2 * np.cos(x_history[:, 3])  # y2

    # Reduce data for animation by skipping frames
    data = data[::FRAME_SKIP]

    # Create the plot
    fig, ax = plt.subplots()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3.1, 3)
    ax.set_aspect('equal')

    # Plot objects
    line, = ax.plot([], [], 'o-', lw=2)  # Pendulum rods
    trace1, = ax.plot([], [], 'r.', markersize=2)  # Trace of the first hinge
    trace2, = ax.plot([], [], 'r.', markersize=2)  # Trace of the pendulum

    # Lists to store trace points
    trace_x1, trace_y1, trace_x2, trace_y2 = [], [], [], []

    def init():
        """
        Initialize the animation by clearing the data.
        """
        line.set_data([], [])
        trace1.set_data([], [])
        trace2.set_data([], [])
        return line, trace1, trace2

    def update(frame):
        """
        Update the animation for the given frame.

        Parameters:
            frame (int): The index of the current frame.
        """
        x1, y1, x2, y2 = data[frame]

        # Set pendulum line positions
        line.set_data([0, x1, x2], [0, y1, y2])

        # Append the trace points
        trace_x1.append(x1)
        trace_y1.append(y1)
        trace_x2.append(x2)
        trace_y2.append(y2)

        # Limit trace size for better performance
        if len(trace_x1) >= 100:
            trace_x1.pop(0)
            trace_y1.pop(0)
        if len(trace_x2) >= 1000:
            trace_x2.pop(0)
            trace_y2.pop(0)

        # Update trace data
        trace1.set_data(trace_x1, trace_y1)
        trace2.set_data(trace_x2, trace_y2)
        return line, trace1, trace2

    num_frames = data.shape[0]

    # Create the animation
    ani = FuncAnimation(
        fig, update,
        frames=num_frames,
        init_func=init,
        blit=True,
        interval=1
    )

    plt.show()
