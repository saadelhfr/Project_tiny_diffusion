import numpy as np
import matplotlib.pyplot as plt
from src.utils.datasets import get_dataset


def plot_test_fn():
    plt.figure(figsize=(20, 10))
    for i in range(0, 10):
        # Load the dataset for the digit 'i'
        data_set = get_dataset(name="mnist", n=10000, digit=i)

        # Convert the dataset to a numpy array
        data_np = data_set.tensors[0].detach().numpy()

        # Determine the subplot index (5 rows, 2 columns)
        ax = plt.subplot(2, 5, i + 1)

        # Create scatter plot for the digit
        ax.scatter(data_np[:, 0], data_np[:, 1], s=1)
        ax.set_title(f"Digit {i}")
        ax.invert_yaxis()

        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])  # Hide x-axis ticks
        ax.set_yticks([])  # Hide y-axis ticks

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_scatter(axis_for_plot, title: str, input_array: np.ndarray):
    """The provided numpy array should be a 2D array with shape (N, 2)"""
    x_min, x_max, y_min, y_max = -1, 1, -1, 1

    axis_for_plot.set_xlim(x_min, x_max)
    axis_for_plot.set_ylim(y_min, y_max)
    axis_for_plot.scatter(input_array[:, 0], input_array[:, 1], s=1)
    axis_for_plot.set_title(title)
    axis_for_plot.set_aspect("equal", adjustable="box")
    axis_for_plot.set_xticks([])  # Hide x-axis ticks
    axis_for_plot.set_yticks([])  # Hide y-axis ticks
    axis_for_plot.invert_yaxis()

    return axis_for_plot


def plot_image(input_array: np.ndarray, axis_for_plot):
    """The provided numpy array should be a 2D array with shape (N, 2)"""
    axis_for_plot.imshow(input_array, cmap="gray")
    axis_for_plot.set_xticks([])  # Hide x-axis ticks
    axis_for_plot.set_yticks([])  # Hide y-axis ticks
    return axis_for_plot
