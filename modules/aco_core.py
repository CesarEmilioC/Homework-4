"""
Core ACO implementation for (A)TSP.

Basic Ant System with:
- probabilistic tour construction
- global pheromone update
- elitist reinforcement
"""

import numpy as np


def run_aco_tsp(
    distance_matrix,
    n_ants=20,
    n_iterations=100,
    alpha=1.0,
    beta=2.0,
    rho=0.5,
    q=1.0,
    seed=None,
):
    """
    Run an Ant System (ACO) for (A)TSP.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Cost/distance matrix (n x n).
    n_ants : int
        Number of ants.
    n_iterations : int
        Number of iterations.
    alpha : float
        Pheromone influence.
    beta : float
        Heuristic influence (1 / distance).
    rho : float
        Pheromone evaporation rate in (0, 1].
    q : float
        Pheromone deposit factor (q / L_k).
    seed : int or None
        Random seed.

    Returns
    -------
    best_tour : list[int]
        Best tour found.
    best_length : float
        Length of the best tour.
    best_history : list[float]
        Best-so-far length at each iteration.
    """
    rng = np.random.default_rng(seed)
    n = distance_matrix.shape[0]

    # -----------------------------------------------------
    # Heuristic matrix: careful handling of zero-cost arcs
    # -----------------------------------------------------
    eta = np.zeros_like(distance_matrix, dtype=float)

    positive_mask = distance_matrix > 0.0
    eta[positive_mask] = 1.0 / distance_matrix[positive_mask]

    # For zero-cost off-diagonal arcs, assign large attractiveness
    offdiag_mask = ~np.eye(n, dtype=bool)
    zero_mask = (distance_matrix == 0.0) & offdiag_mask

    if np.any(positive_mask):
        max_eta = eta[positive_mask].max()
        eta[zero_mask] = 10.0 * max_eta
    else:
        eta[zero_mask] = 1.0

    # ---------------------------------------------
    # Initial pheromone matrix
    # ---------------------------------------------
    tau = np.full((n, n), 1.0, dtype=float)

    best_tour = None
    best_length = float("inf")
    best_history = []

    # =====================================================
    # Main Optimization Loop
    # =====================================================
    for it in range(n_iterations):
        all_tours = []
        all_lengths = []

        # ---------------------------------------------
        # Construct tours
        # ---------------------------------------------
        for _ in range(n_ants):

            start = rng.integers(0, n)
            tour = [start]

            unvisited = set(range(n))
            unvisited.remove(start)
            current = start

            while unvisited:
                cities = list(unvisited)
                probs = []

                # Transition probabilities
                for j in cities:
                    tau_ij = tau[current, j] ** alpha
                    eta_ij = eta[current, j] ** beta
                    probs.append(tau_ij * eta_ij)

                probs = np.array(probs, dtype=float)
                if probs.sum() == 0.0:
                    probs[:] = 1.0 / len(probs)
                else:
                    probs /= probs.sum()

                next_city = rng.choice(cities, p=probs)
                tour.append(next_city)
                unvisited.remove(next_city)
                current = next_city

            length = tour_length(tour, distance_matrix)

            all_tours.append(tour)
            all_lengths.append(length)

            # Update global best
            if length < best_length:
                best_length = length
                best_tour = tour

        best_history.append(best_length)

        # ---------------------------------------------
        # Pheromone update
        # ---------------------------------------------
        tau *= (1.0 - rho)

        # Deposit pheromone from all tours
        for tour, length in zip(all_tours, all_lengths):
            deposit = q / length
            for i in range(len(tour) - 1):
                a, b = tour[i], tour[i + 1]
                tau[a, b] += deposit
            tau[tour[-1], tour[0]] += deposit

        # Elitist reinforcement
        if best_tour is not None:
            elitist_factor = 2.0
            deposit_best = elitist_factor * (q / best_length)
            for i in range(len(best_tour) - 1):
                a, b = best_tour[i], best_tour[i + 1]
                tau[a, b] += deposit_best
            tau[best_tour[-1], best_tour[0]] += deposit_best

    return best_tour, best_length, best_history


def tour_length(tour, distance_matrix):
    """
    Compute the total length of a tour, including return to start.
    """
    length = 0.0
    for i in range(len(tour) - 1):
        length += distance_matrix[tour[i], tour[i + 1]]
    length += distance_matrix[tour[-1], tour[0]]
    return length