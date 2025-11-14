"""
Utility functions for reading TSPLIB ATSP/TSP instances
and working with tours.

This module keeps the parsing minimal and focused on what is required
for the assignment: reading FULL_MATRIX ATSP/TSP files and extracting
the distance matrix.
"""

import numpy as np


def read_tsplib(path):
    """
    Read a TSPLIB (A)TSP instance and extract the distance matrix.

    The parser is lightweight and assumes:
    - DIMENSION is provided,
    - EDGE_WEIGHT_SECTION follows the header,
    - EDGE_WEIGHT_FORMAT = FULL_MATRIX (typical for ATSP files),
    - all weights appear in row-major order.

    Parameters
    ----------
    path : str
        Path to a TSPLIB file (e.g., "instances/br17.atsp").

    Returns
    -------
    dict
        Contains:
        - 'name'              : problem name
        - 'dimension'         : number of cities
        - 'edge_weight_type'  : as read from file
        - 'edge_weight_format': as read from file
        - 'distance_matrix'   : (n x n) float array
    """
    # Read all non-empty lines
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    name = None
    dimension = None
    edge_weight_type = None
    edge_weight_format = None
    weights_start_idx = None

    # Parse header
    for i, line in enumerate(lines):
        upper = line.upper()

        if upper.startswith("NAME"):
            # Accept formats "NAME: br17" or "NAME br17"
            name = line.split(":", 1)[1].strip() if ":" in line else line.split()[1]

        elif upper.startswith("DIMENSION"):
            dimension = int(line.split(":", 1)[1].strip())

        elif upper.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":", 1)[1].strip()

        elif upper.startswith("EDGE_WEIGHT_FORMAT"):
            edge_weight_format = line.split(":", 1)[1].strip()

        elif upper.startswith("EDGE_WEIGHT_SECTION"):
            # The next line is the start of the numeric matrix
            weights_start_idx = i + 1
            break

    if dimension is None or weights_start_idx is None:
        raise ValueError("Could not parse DIMENSION or EDGE_WEIGHT_SECTION from TSPLIB file.")

    # Collect all numbers following the EDGE_WEIGHT_SECTION
    numbers = []
    for line in lines[weights_start_idx:]:
        if line.upper().startswith("EOF"):
            break
        parts = line.split()
        numbers.extend(float(p) for p in parts)

    # TSPLIB FULL_MATRIX format requires n*n values
    expected = dimension * dimension
    if len(numbers) != expected:
        raise ValueError(
            f"Expected {expected} weights for FULL_MATRIX, got {len(numbers)}. "
            "Check the TSPLIB file format."
        )

    matrix = np.array(numbers, dtype=float).reshape((dimension, dimension))

    return {
        "name": name,
        "dimension": dimension,
        "edge_weight_type": edge_weight_type,
        "edge_weight_format": edge_weight_format,
        "distance_matrix": matrix,
    }


def tour_length(tour, distance_matrix):
    """
    Compute the total length of a TSP/ATSP tour.

    The tour is assumed to be a permutation of cities, and the path
    returns to the starting city.

    Parameters
    ----------
    tour : sequence[int]
        Tour as a list or tuple of city indices.
    distance_matrix : np.ndarray
        Cost matrix.

    Returns
    -------
    float
        Total cost of the tour.
    """
    length = 0.0
    n = len(tour)
    for i in range(n - 1):
        length += distance_matrix[tour[i], tour[i + 1]]
    length += distance_matrix[tour[-1], tour[0]]  # close the cycle
    return length