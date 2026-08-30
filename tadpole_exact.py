"""
Top-level driver, rewritten to match the author's actual search structure:
exactly r+1 global configurations for r pendant paths (all gap k, or one
path bumped to gap k+1), each giving a single deterministic placement per
path -- not an independent per-path candidate product.
"""

from tadpole_core import (
    TadpoleGraph, is_k_truncated_resolving, brute_force_min_resolving,
    greedy_repair,
)
from pendant_placement import path_placement, global_configurations
from sweep_setup import build_initial_active_groups
from sweep_loop import sweep_from, sweep_from_clean_frontier




def alternating_cycle_gap_candidates(graph):
    """Construct the special all-(3k+2) cycle patterns described by the author.

    Consecutive cycle-resolver gaps alternate k, 2k, k, 2k, ... .  When
    divisibility requires it, one of the 2k gaps may be enlarged to 2k+1.
    Every rotation and both alternating phases are returned.  ``gap`` means
    the number of cycle vertices strictly between consecutive resolvers.
    """
    n, k = graph.n, graph.k
    out = set()
    for q in range(2, n + 1):
        for phase in (0, 1):
            base = [k if (i + phase) % 2 == 0 else 2 * k for i in range(q)]
            variants = [base]
            for i, g in enumerate(base):
                if g == 2 * k:
                    v = list(base)
                    v[i] = 2 * k + 1
                    variants.append(v)
            for gaps in variants:
                if sum(g + 1 for g in gaps) != n:
                    continue
                for start in range(n):
                    verts = [start]
                    cur = start
                    ok = True
                    for g in gaps[:-1]:
                        cur = (cur + g + 1) % n
                        if cur in verts:
                            ok = False
                            break
                        verts.append(cur)
                    if ok and (cur + gaps[-1] + 1) % n == start:
                        out.add(frozenset(verts))
    return [set(x) for x in out]


def no_distant_pendant_vertex(graph, dist, resolving_set):
    """Whether the current path-side choice leaves no pendant vertex distant.

    A pendant vertex is distant when its k-truncated distance is k+1 from
    every currently selected resolving vertex.
    """
    if not resolving_set:
        return False
    cutoff = graph.k + 1
    for w in graph.all_vertices[graph.n:]:
        if w in resolving_set:
            continue
        if all(dist[s][w] >= cutoff for s in resolving_set):
            return False
    return True



def unresolved_pair_start_candidates(graph, dist, resolving_set):
    """Return cycle start candidates from one unresolved pair.

    Pick the first pair with identical k-truncated representations under the
    current resolving set. Any additional resolver distinguishing the pair must
    lie within distance k of at least one endpoint. We therefore start sweeps
    only from the UNION of the two endpoints' cycle k-neighbourhoods.

    For a pendant-path endpoint, we project its k-neighbourhood onto cycle
    vertices simply by testing actual graph distance to every cycle vertex.
    """
    k = graph.k
    vertices = list(graph.all_vertices)

    signatures = {}
    for v in vertices:
        sig = tuple(min(dist[s][v], k + 1) for s in resolving_set)
        if sig in signatures:
            u = signatures[sig]
            cand = {
                y for y in range(graph.n)
                if dist[y][u] <= k or dist[y][v] <= k
            }
            return cand
        signatures[sig] = v

    return set()


def solve_for_config(graph, dist, gaps):
    """gaps: list of assigned terminal gaps, one per path (k or k+1)."""
    k = graph.k
    placements = {}
    for j, (attach, m) in enumerate(graph.pendant_paths):
        placements[j] = path_placement(m, k, gaps[j])

    S_P = set()
    for j, placement in placements.items():
        for local_idx in placement['indices']:
            S_P.add(graph.path_vertices[j][local_idx])

    # Collapsed near-join vertices are mandatory but are NOT admissible
    # intervals offering a choice of position -- pre-place them (like
    # S_P) rather than running them through the active-group / starting-
    # interval selection machinery.
    collapsed_vertices = {
        graph.join_vertex(j) for j, placement in placements.items()
        if placement.get('collapsed')
    }
    S_P_effective = S_P | collapsed_vertices

    # Any OTHER admissible interval that reduces to a forced singleton
    # (e.g. a short path's own Claim domain with a=k+1, undischarged) is
    # equally mandatory and equally not a real "choice" -- generalize the
    # pre-placement above to any such group, iterating since pre-placing
    # one can newly discharge (or newly force) others. This lets the
    # existing, already-validated discharge machinery -- not a new ad-hoc
    # distance rule -- determine what a forced vertex's presence resolves.
    forced_vertices = set()
    coverage_arcs = []
    while True:
        groups, coverage_arcs = build_initial_active_groups(graph, dist, placements, S_P_effective)
        newly_forced = {
            next(iter(g['candidates'][0])) for g in groups
            if len(g['candidates']) == 1 and len(g['candidates'][0]) == 1
        } - forced_vertices
        if not newly_forced:
            break
        forced_vertices |= newly_forced
        S_P_effective = S_P_effective | forced_vertices

    collapsed_vertices = collapsed_vertices | forced_vertices
    # drop the now-redundant singleton groups themselves (they're
    # pre-placed, not something the sweep needs to select)
    groups = [
        g for g in groups
        if not (len(g['candidates']) == 1 and len(g['candidates'][0]) == 1
                and next(iter(g['candidates'][0])) in forced_vertices)
    ]
    # Pendant-derived active groups are the persistent constraints.
    # Large gaps are NOT converted to static active intervals; the frontier
    # sweep generates its temporary coverage interval dynamically.
    clean_groups = [
        {**g, 'candidates': [set(s) for s in g['candidates']]}
        for g in groups
    ]

    # The pendant/collapsed/forced vertices may already resolve the whole
    # graph. In that case no cycle sweep is needed at all.
    if is_k_truncated_resolving(
        graph, S_P | collapsed_vertices, dist
    ):
        return S_P, collapsed_vertices

    # Forced singleton cycle vertices are mandatory placements and therefore
    # natural sweep origins. Collapsed path resolvers are also preselected,
    # but a collapse by itself does not determine the sweep origin.
    forced_singleton_vertices = set(forced_vertices)

    if forced_singleton_vertices:
        start_candidates = set(forced_singleton_vertices)
    elif clean_groups:
        # Otherwise start from one minimum-cardinality active admissible group.
        # For an OR group, cardinality is the size of the union of its branches.
        def group_cardinality(g):
            return len(set().union(*g['candidates']))

        start_group = min(clean_groups, key=group_cardinality)
        start_candidates = set().union(*start_group['candidates'])
    else:
        # No pendant-derived admissible interval remains, but the current set
        # is not yet resolving (the early-exit check above already handled the
        # resolving case). Choose one unresolved pair and use only the UNION of
        # the two endpoints' cycle k-neighbourhoods as possible optimal anchors.
        start_candidates = unresolved_pair_start_candidates(
            graph, dist, S_P | collapsed_vertices
        )
        if not start_candidates:
            # Defensive fallback: this should be unreachable because the
            # current set was already checked to be non-resolving.
            start_candidates = {0}

    best_S_C = None
    best_valid = False
    best_union_size = None
    for start_vertex in start_candidates:
        # The only branching in the simple sweep concerns requirements tied
        # to the starting cut: initial OR/symmetry may be handled immediately
        # or deferred to wraparound. Retain the smaller valid result.
        candidates_for_start = [
            sweep_from_clean_frontier(
                graph, dist, clean_groups, start_vertex, S_P_effective,
                placements=placements, defer_initial_special=False
            ),
            sweep_from_clean_frontier(
                graph, dist, clean_groups, start_vertex, S_P_effective,
                placements=placements, defer_initial_special=True
            ),
        ]

        for S_C in candidates_for_start:
            candidate_full = S_P | collapsed_vertices | S_C
            valid = is_k_truncated_resolving(graph, candidate_full, dist)
            union_size = len(collapsed_vertices | S_C)
            if best_S_C is None:
                best_S_C, best_valid, best_union_size = S_C, valid, union_size
            elif valid and not best_valid:
                best_S_C, best_valid, best_union_size = S_C, valid, union_size
            elif valid == best_valid and union_size < best_union_size:
                best_S_C, best_valid, best_union_size = S_C, valid, union_size

    # Special configuration: if every pendant path has order 3k+2 and
    # the chosen path-side resolvers leave no distant pendant vertex, also
    # try the author's explicit alternating k / 2k cycle-gap construction
    # (with one 2k+1 gap when needed).  This is a structural construction,
    # not a brute-force repair; final distance validation merely guards the
    # implementation against a malformed pattern.
    if (all(m == 3 * k + 2 for _, m in graph.pendant_paths)
            and no_distant_pendant_vertex(graph, dist, S_P_effective)):
        for special_C in alternating_cycle_gap_candidates(graph):
            candidate_full = S_P | collapsed_vertices | special_C
            if not is_k_truncated_resolving(graph, candidate_full, dist):
                continue
            union_size = len(collapsed_vertices | special_C)
            if (best_S_C is None or not best_valid
                    or union_size < best_union_size):
                best_S_C = special_C
                best_valid = True
                best_union_size = union_size

    return S_P, collapsed_vertices | best_S_C


def solve_exact(graph, allow_brute_fallback=True, brute_force_size_limit=14,
                 verbose=False):
    dist = graph.all_pairs_dist()
    k = graph.k

    for attach, length in graph.pendant_paths:
        if not (k + 1 <= length <= 3 * k + 2):
            raise ValueError(
                f"pendant path at {attach} has order {length}, outside "
                f"the restricted family [k+1, 3k+2]"
            )

    path_orders = [m for (_, m) in graph.pendant_paths]
    configs = global_configurations(path_orders, k)

    best_S, best_info = None, None

    for gaps in configs:
        S_P, S_C = solve_for_config(graph, dist, gaps)
        S = S_P | S_C
        valid = is_k_truncated_resolving(graph, S, dist)
        repaired = False

        if not valid:
            if allow_brute_fallback and len(graph.all_vertices) <= brute_force_size_limit:
                S = brute_force_min_resolving(graph, dist=dist)
                if S is None:
                    continue
                repaired = True
            else:
                S = greedy_repair(graph, S, dist=dist)
                repaired = True
            valid = is_k_truncated_resolving(graph, S, dist)

        if not valid:
            continue

        if verbose:
            print(f"gaps={gaps} -> |S_P|={len(S_P)} |S_C|={len(S_C)} "
                  f"total={len(S)} repaired={repaired}")

        if best_S is None or len(S) < len(best_S):
            best_S = S
            best_info = {'S_P': S_P, 'S_C': S_C, 'repaired': repaired, 'gaps': gaps}

    if best_S is None:
        if allow_brute_fallback and len(graph.all_vertices) <= brute_force_size_limit:
            best_S = brute_force_min_resolving(graph, dist=dist)
            best_info = {'repaired': True, 'brute_force_only': True}
        else:
            raise RuntimeError("no valid resolving set found and graph too large for fallback")

    return best_S, best_info


if __name__ == "__main__":
    g = TadpoleGraph(n=39, pendant_paths=[(5, 5), (7,4), (18, 5), (20,3), (31, 5), (33,3)], k=2)
    S, info = solve_exact(g, verbose=True)
    print("Resolving set:", sorted(S), "size:", len(S))
    print("Valid:", is_k_truncated_resolving(g, S))
    print(info['repaired'])
