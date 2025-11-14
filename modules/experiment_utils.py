"""
Helper functions to run multiple ACO experiments and compute basic statistics.

These utilities wrap the ACO solver so that:
- multiple independent runs can be executed easily,
- aggregated statistics (mean, std, etc.) can be computed,
- statistical tests (Wilcoxon rank-sum) can be applied when SciPy is available.
"""

import numpy as np

try:
    # SciPy is optional in this project, so we guard the import
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

from .aco_core import run_aco_tsp


def run_multiple_experiments(
    distance_matrix,
    instance_name,
    config_name,
    n_runs,
    aco_params,
    base_seed=0,
):
    """
    Execute several independent ACO runs for the same instance and configuration.

    Each run uses a different seed so that the results are statistically independent.
    Only the best solution from each run is kept.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Cost matrix defining the TSP/ATSP instance.
    instance_name : str
        Name of the current instance (e.g., 'br17').
    config_name : str
        Configuration label (e.g., 'config_1').
    n_runs : int
        Number of independent executions.
    aco_params : dict
        Parameters forwarded directly to run_aco_tsp.
    base_seed : int
        Base value for reproducible seeding. Run i uses seed = base_seed + i.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'instance'
            - 'config'
            - 'lengths' : list of best lengths over the runs
            - 'tours'   : list of best tours
    """
    lengths = []
    tours = []

    # Launch n_runs independent ACO executions
    for run_id in range(n_runs):
        seed = base_seed + run_id
        best_tour, best_length, _ = run_aco_tsp(
            distance_matrix=distance_matrix,
            seed=seed,
            **aco_params,
        )
        lengths.append(best_length)
        tours.append(best_tour)

    return {
        "instance": instance_name,
        "config": config_name,
        "lengths": lengths,
        "tours": tours,
    }


def summarize_results(lengths, best_known=None):
    """
    Compute simple descriptive statistics for a collection of values.

    Parameters
    ----------
    lengths : list[float] or np.ndarray
        List of objective values (one per run).
    best_known : float or None
        If provided, counts how many runs reached this value exactly.

    Returns
    -------
    dict
        Contains the fields:
            n, mean, median, std, min, max,
            and optionally 'hits_best_known' if best_known is provided.
    """
    arr = np.asarray(lengths, dtype=float)

    summary = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)),  # sample standard deviation
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

    if best_known is not None:
        hits = int(np.sum(arr == float(best_known)))
        summary["hits_best_known"] = hits

    return summary


def wilcoxon_rank_sum(sample_a, sample_b):
    """
    Apply the Wilcoxon rank-sum (Mann–Whitney U) test to compare two samples.

    This function simply wraps SciPy's implementation to keep the code clean.

    Parameters
    ----------
    sample_a : array-like
        First sample of results.
    sample_b : array-like
        Second sample of results.

    Returns
    -------
    dict
        Contains 'statistic' and 'p_value'.

    Raises
    ------
    ImportError
        If SciPy is not installed.
    """
    if stats is None:
        raise ImportError(
            "SciPy is required to run the Wilcoxon rank-sum test. "
            "Install it with `pip install scipy`."
        )

    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)

    # Classical rank-sum test for independent samples
    stat, p = stats.ranksums(a, b)
    return {"statistic": float(stat), "p_value": float(p)}