import numpy as np
import pandas as pd


def path_lengths(path: pd.DataFrame) -> np.ndarray:
    """Compute the path lengths of a trajectory

    Parameters:
        path (pd.DataFrame): The trajectory

    Returns:
        np.ndarray: Path length
    """
    # Get the difference vectors
    differences = np.diff(path, axis=0)

    # Get the length of each vector, aka the Euclidean distance between each point.
    return np.linalg.norm(differences, axis=1)


def mean_squared_displacement(path: pd.DataFrame) -> np.ndarray:
    """Compute the mean-squared displacement of a trajectory

    Parameters:
        path (pd.DataFrame): The trajectory

    Returns:
        np.ndarray: Mean squared displacements
    """
    # Get the number of steps
    steps = path.shape[0]

    # Initialize the result
    results = np.zeros(steps - 1)

    # We start from 1 and end with steps - 1 displacements
    for displacement in range(1, steps):
        # Get the displacements. Here, we take the last elements after "n" displacements and do the
        # difference between the first elements until "N" - "n" displacements
        displacements = path.values[displacement:] - path.values[:-displacement]

        # Every magnitude vector pwo to 2.
        displacements **= 2

        # Here we handle the dimensions' problem, every displacement is sumed over each component (x + y)
        displacements_1d = np.sum(displacements, axis=1)

        # Get the mean of the displacements
        results[displacement - 1] = np.mean(displacements_1d)

    return results


def turning_angles(path: pd.DataFrame) -> np.ndarray:
    """Compute the turning angles of a trajectory

    Parameters:
        path (pd.DataFrame): The trajectory

    Returns:
        np.ndarray: The turning angles in radians
    """
    # Compute the directions for each pair of points
    vectors = np.diff(path, axis=0)

    # Create two arrays that match each pair of vectors
    a, b = vectors[:-1], vectors[1:]

    # Create an array of amtrices that will be used to get the determinant of each pair of vectors
    matrices = np.stack((a, b), axis=1)

    # Just calculates the angles
    return np.atan2(
        np.linalg.det(matrices),
        np.vecdot(a, b),
    )
