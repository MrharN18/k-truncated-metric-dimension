
"""Cycle-only exhaustive oracle for a fixed pendant-path resolving set S_P.

This module evaluates only the cycle-placement part of the algorithm.
The pendant-path vertices S_P are held fixed, and the oracle searches
subsets of V(C_n) for a minimum S_C such that S_P ∪ S_C is a
k-truncated resolving set.

This does NOT test whether the chosen pendant-path configuration itself is
globally optimal.
"""

from itertools import combinations

from tadpole_core import is_k_truncated_resolving


def brute_force_min_cycle_resolving(
    graph,
    S_P,
    dist=None,
    max_cycle_size=None,
):
    """Return a minimum cycle set S_C for the fixed path set S_P.

    Parameters
    ----------
    graph : TadpoleGraph
        The graph being solved.
    S_P : iterable of int
        Fixed resolving vertices on pendant paths. The oracle does not alter
        this set.
    dist : dict, optional
        Precomputed all-pairs distances.
    max_cycle_size : int, optional
        Search only cycle sets of size at most this value. This is useful when
        the algorithm's |S_C| is already known: if no smaller set exists, the
        algorithmic cycle placement is optimal for this fixed S_P.

    Returns
    -------
    set[int] or None
        A minimum S_C subset of {0,...,n-1}, or None if no solution exists
        within max_cycle_size.
    """
    if dist is None:
        dist = graph.all_pairs_dist()

    S_P = set(S_P)
    cycle_vertices = list(range(graph.n))

    # It is possible that S_P alone already resolves the graph.
    if is_k_truncated_resolving(graph, S_P, dist):
        return set()

    if max_cycle_size is None:
        max_cycle_size = graph.n
    else:
        max_cycle_size = min(max_cycle_size, graph.n)

    for size in range(1, max_cycle_size + 1):
        for C in combinations(cycle_vertices, size):
            S_C = set(C)
            if is_k_truncated_resolving(graph, S_P | S_C, dist):
                return S_C

    return None


def cycle_sweep_is_optimal_for_fixed_SP(
    graph,
    S_P,
    S_C_algorithm,
    dist=None,
):
    """Compare an algorithmic cycle set with the cycle-only exhaustive oracle.

    The search stops at |S_C_algorithm|.  The returned dictionary reports the
    exact optimum for the fixed S_P and the additive gap of the algorithm.
    """
    if dist is None:
        dist = graph.all_pairs_dist()

    S_P = set(S_P)
    S_C_algorithm = set(S_C_algorithm)

    oracle = brute_force_min_cycle_resolving(
        graph,
        S_P,
        dist=dist,
        max_cycle_size=len(S_C_algorithm),
    )

    if oracle is None:
        # If the supplied algorithmic set is valid this should not happen.
        return {
            "valid_algorithm": is_k_truncated_resolving(
                graph, S_P | S_C_algorithm, dist
            ),
            "algorithm_size": len(S_C_algorithm),
            "oracle_size": None,
            "oracle_set": None,
            "additive_gap": None,
            "optimal_for_fixed_SP": False,
        }

    alg_valid = is_k_truncated_resolving(
        graph, S_P | S_C_algorithm, dist
    )
    return {
        "valid_algorithm": alg_valid,
        "algorithm_size": len(S_C_algorithm),
        "oracle_size": len(oracle),
        "oracle_set": oracle,
        "additive_gap": len(S_C_algorithm) - len(oracle),
        "optimal_for_fixed_SP": alg_valid and len(S_C_algorithm) == len(oracle),
    }
