import numpy as np
import pandas as pd
from scipy.stats import wrapcauchy, levy_stable


def brownian_motion_2d(
    steps: int,
    length: float = 1.0,
    start: tuple = (0, 0),
) -> pd.DataFrame:
    """Generate a 2D Brownian motion trajectory

    Parameters:
        steps (int): Number of steps
        length (float, optional): Each step length
        start (tuple, optional): Starting position

    Returns:
        pd.DataFrame: 2D Brownian motion trajectory
    """
    # Choose random angles randomly between (0, 90, 180, 270) degrees in radians.
    angles = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2], size=steps - 1)

    # Cumulative angles
    cumulative_angles = np.cumsum(angles)

    # Velocities with a length of 1.0
    velocities = np.column_stack([np.cos(cumulative_angles), np.sin(cumulative_angles)])

    # We adjust vectors' length
    velocities *= length

    # Prepend the initial position
    velocities = np.vstack([start, velocities])

    # The cumulative sum of the velocities
    positions = np.cumsum(velocities, axis=0)

    # Wrap the positions in a pandas DataFrame
    return pd.DataFrame(positions, columns=["x", "y"])


def correlated_random_walk_2d(
    steps: int,
    length: float = 1.0,
    start: tuple = (0, 0),
    c: float = 0.5,
) -> pd.DataFrame:
    """
    Generate a 2D correlated random walk trajectory.

    Parameters:
        steps (int): Number of steps
        length (float, optional): Each step length
        start (tuple, optional): Starting position
        c (float, optional): Correlation coefficient

    Returns:
        pd.DataFrame: 2D Brownian motion trajectory
    """
    # Choose random angles randomly between (0, 90, 180, 270) degrees in radians.
    angles = wrapcauchy.rvs(c=c, size=steps - 1)

    # Cumulative angles
    cumulative_angles = np.cumsum(angles)

    # Velocities with a length of 1.0
    velocities = np.column_stack([np.cos(cumulative_angles), np.sin(cumulative_angles)])

    # We adjust vectors' length
    velocities *= length

    # Prepend the initial position
    velocities = np.vstack([start, velocities])

    # The cumulative sum of the velocities
    positions = np.cumsum(velocities, axis=0)

    # Wrap the positions in a pandas DataFrame
    return pd.DataFrame(positions, columns=["x", "y"])


def levy_walk_2d(
    steps: int,
    start: tuple = (0, 0),
    c: float = 0.5,
    alpha: float = 1.0,
    beta: float = 0.0,
    m: float = 3.0,
) -> pd.DataFrame:
    """Generate a 2D Lévy flight trajectory.

    Parameters:
        steps (int): Number of steps
        start (tuple, optional): Starting position
        c (float, optional): Correlation coefficient
        alpha (float, optional): Alpha
        beta (float, optional): Beta
        m (float, optional): M

    Returns:
        pd.DataFrame: 2D Levy walk trajectory
    """
    # Choose random angles randomly between a wrapped Cauchy distribution
    angles = wrapcauchy.rvs(c=c, size=steps - 1)

    # Cumulative angles
    cumulative_angles = np.cumsum(angles)

    # Velocities with a length of 1.0
    velocities = np.column_stack([np.cos(cumulative_angles), np.sin(cumulative_angles)])

    # Choose random step_lengths from a Levy distribution
    lengths = levy_stable.rvs(alpha=alpha, beta=beta, loc=m, size=steps - 1)

    # The reshaping is needed before, so we can do an element-wise multiplication
    # between our array of velocities and the lengths.
    velocities *= lengths.reshape(-1, 1)

    # Prepend the initial position
    velocities = np.vstack([start, velocities])

    # The cumulative sum of the velocities
    positions = np.cumsum(velocities, axis=0)

    # Wrap the positions in a pandas DataFrame
    return pd.DataFrame(positions, columns=["x", "y"])
