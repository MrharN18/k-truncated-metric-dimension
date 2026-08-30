"""
The directional sweep loop itself.
"""

from sweep_setup import (
    residual_of, current_radius, distant_vertex_exists, cycle_ball,
    candidates_for_targets, admissible_target_residual,
)
from tadpole_core import k_trunc, is_k_truncated_resolving


def rel_factory(n, start):
    return lambda v: (v - start) % n


def group_earliest(group, rel):
    return min(rel(v) for s in group['candidates'] for v in s)


def group_has_vertex_at_or_after(group, rel, pos_rel):
    """True if some branch of this OR-group offers a vertex at or after
    pos_rel in sweep order -- i.e. the group can still be satisfied
    without backtracking, even if a DIFFERENT branch lies behind pos."""
    return any(rel(v) >= pos_rel for s in group['candidates'] for v in s)


def group_contains(group, v):
    return any(v in s for s in group['candidates'])


def update_group_after_placement(graph, dist, group, s_t, radius, selected_resolvers=None):
    """Update one active group after selecting ``s_t``.

    Target-carrying admissible residuals are updated from their TARGETS,
    regardless of whether ``s_t`` lies in the old candidate interval: a new
    resolver may handle only part of the target set, which can widen/shift the
    recomputed candidate interval.

    Other group types keep their established behavior.
    """
    if group.get('kind') == 'admissible_residual':
        target_radius=group.get('radius', radius)
        remaining={
            u for u in group.get('targets', set())
            if dist[s_t][u] > target_radius
        }
        if not remaining:
            return None

        cand=candidates_for_targets(
            graph, dist, remaining, target_radius
        )
        if not cand:
            return None

        out=dict(group)
        out['targets']=remaining
        out['candidates']=[cand]
        return out

    branch = None
    for cand_set in group['candidates']:
        if s_t in cand_set:
            branch = cand_set
            break
    if branch is None:
        return group  # untouched
    if group.get('kind') == 'existence':
        return None  # pure existence constraints discharge outright

    if group.get('kind') == 'admissible':
        selected_resolvers = set(selected_resolvers or {s_t})
        additional_resolvers = selected_resolvers - {
            s_t, group['path_resolver']
        }
        return admissible_target_residual(
            graph, dist,
            set(group.get('domain', branch)),
            group['path_resolver'],
            s_t,
            additional_resolvers,
            graph.k,
        )

    res = residual_of(graph, dist, branch, s_t, radius)
    if res is None:
        return None  # discharged
    return ({'candidates': [res], 'kind': 'existence', 'source': ('residual',)}
            if radius == graph.k else {'candidates': [res]})


def fully_distant_vertices(graph, dist, resolving_set):
    """Vertices at truncated distance k+1 from every current resolver."""
    if not resolving_set:
        return []
    k = graph.k
    return [
        v for v in graph.all_vertices
        if v not in resolving_set
        and all(k_trunc(dist[s][v], k) == k + 1 for s in resolving_set)
    ]


def find_symmetric_pairs(graph, dist, s_t, resolving_set_excl_st):
    """Pairs {u,u'} tied through s_t (both <=k from s_t, equal) and tied
    (both k+1) through every other current resolving vertex."""
    k = graph.k
    pairs = []
    close = [v for v in graph.all_vertices
             if k_trunc(dist[s_t][v], k) <= k]
    by_dist = {}
    for v in close:
        by_dist.setdefault(k_trunc(dist[s_t][v], k), []).append(v)
    for d_val, verts in by_dist.items():
        if len(verts) < 2:
            continue
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, up = verts[i], verts[j]
                if all(k_trunc(dist[s][u], k) == k_trunc(dist[s][up], k)
                       for s in resolving_set_excl_st):
                    pairs.append((u, up))
    return pairs


def symmetric_resolution_group(graph, dist, pairs):
    """All cycle vertices y such that placing a resolving vertex at y
    breaks every pair in `pairs` (i.e. y itself distinguishes each pair,
    via ordinary k-truncated distance -- not the distant-vertex budget,
    which is not mentioned for this construct)."""
    k = graph.k
    cand = {
        y for y in range(graph.n)
        if all(k_trunc(dist[y][u], k) != k_trunc(dist[y][up], k) for (u, up) in pairs)
    }
    return {'candidates': [cand], 'kind': 'existence'} if cand else None


def simulate_placement(graph, dist, active, s_t, radius):
    """Apply update_group_after_placement to every group; return the
    resulting active list (discharged groups dropped)."""
    new_active = []
    for g in active:
        ng = update_group_after_placement(graph, dist, g, s_t, radius)
        if ng is not None:
            new_active.append(ng)
    return new_active


def large_gap_group(graph, dist, s_t, gap_vertices, radius):
    cand = {y for y in gap_vertices if radius < dist[s_t][y] <= 2 * radius + 1}
    return {'candidates': [cand]} if cand else None


def eligible_for_distant_check(graph, S_P_global, rel, s_t_rel):
    """Path vertices always count; cycle vertices count only if the sweep
    has already passed their position (rel <= current). A cycle vertex
    the sweep hasn't reached yet may look distant right now, but that's
    not a real budget concern -- just an ordinary outstanding requirement
    still to be addressed."""
    n = graph.n
    return {v for v in graph.all_vertices
            if v >= n or rel(v) <= s_t_rel}


def next_boundary(graph, n, rel, from_rel, active):
    """Nearest active-group vertex position after from_rel, or the
    wraparound sentinel n if none. Only active groups count here --
    pre-placed vertices are handled separately by the chain in
    large_gap_check, since after passing one we can keep checking
    further segments, whereas an active-group boundary is where the
    sweep will naturally pick up next and shouldn't be preempted."""
    next_rel = n
    for g in active:
        for cand_set in g['candidates']:
            for v in cand_set:
                r = rel(v)
                if r > from_rel:
                    next_rel = min(next_rel, r)
    return next_rel


def path_anchor_points(graph, rel, placements, current_full):
    """Return pendant resolving vertices as forward sweep boundaries.

    A pendant resolver is not located at its join vertex for gap purposes.
    Its effective forward position is

        rel(join_vertex) + a,

    where ``a`` is its real distance from the join.  This is exactly the
    comparison needed when deciding whether the next closest resolving
    boundary is a path resolver or the wraparound/start boundary.  Anchors
    whose effective position is beyond one full turn are deliberately
    omitted: the sweep reaches the start vertex first.

    Tuples are ``(effective_rel, actual_resolver, join_rel)``.  The actual
    resolver is retained because distance/radius checks must use the real
    graph vertex, while ``effective_rel`` is used only for ordering.
    """
    if placements is None:
        return []
    n = graph.n
    out = []
    for j, placement in placements.items():
        if placement.get('collapsed'):
            continue
        indices = placement.get('indices', [])
        if not indices:
            continue
        resolver = graph.path_vertices[j][indices[0]]  # nearest-to-join resolver
        if resolver not in current_full:
            continue
        a = placement.get('a', 0)
        eff = rel(graph.join_vertex(j)) + a
        if 0 < eff <= n:
            out.append((eff, resolver, rel(graph.join_vertex(j))))
    return sorted(out, key=lambda x: x[0])


def _legacy_large_gap_check(graph, dist, k, n, rel, start_vertex, s_t, active, current_full,
                            placements=None):
    """Walk forward from s_t through consecutive boundary-to-boundary
    segments, checking each for the large-gap threshold, continuing past
    pre-placed/already-swept cycle vertices (each becomes the new anchor
    for the next segment) but stopping at the first active-group boundary
    (that segment is the sweep's own to handle next) or the full
    wraparound back to start_vertex. Returns a new group to add, or None.
    `current_full` is the full current resolving set (S_P_global plus
    everything placed by the sweep so far, including s_t itself)."""
    anchor = s_t
    anchor_rel = rel(s_t)
    active_boundary_rel = next_boundary(graph, n, rel, anchor_rel, active)

    # Existing cycle resolvers use their literal sweep positions. Pendant
    # resolvers use their effective positions rel(join)+a.  Comparing these
    # in one ordered list is the unfinished "closest resolving vertices"
    # fix discussed with the author.
    boundaries = []
    for v in current_full:
        if v < n and v != s_t:
            r = rel(v)
            if r > anchor_rel:
                boundaries.append((r, v, 'cycle'))
    for eff, resolver, join_rel in path_anchor_points(
            graph, rel, placements, current_full):
        if eff > anchor_rel and resolver != s_t:
            boundaries.append((eff, resolver, 'path'))
    boundaries.sort(key=lambda x: x[0])

    idx = 0
    while True:
        next_known_rel = boundaries[idx][0] if idx < len(boundaries) else n
        boundary_rel = min(active_boundary_rel, next_known_rel)

        gap_start = anchor_rel + 1
        gap_end = boundary_rel
        gap_len = gap_end - gap_start
        eligible = eligible_for_distant_check(graph, current_full, rel,
                                               min(anchor_rel, n - 1))
        radius = current_radius(graph, dist, current_full, eligible)
        threshold = 2 * k + 1 if radius == k + 1 else 2 * k
        if gap_len > threshold:
            # Only real cycle positions can receive a newly created cycle
            # resolver.  Effective path-depth coordinates beyond n-1 are
            # therefore clipped out of this conversion.
            real_start = min(gap_start, n)
            real_end = min(gap_end, n)
            gap_vertices = {
                (start_vertex + p) % n for p in range(real_start, real_end)
            }
            lg = large_gap_group(graph, dist, anchor, gap_vertices, radius)
            if lg is not None:
                return lg

        if boundary_rel == active_boundary_rel:
            return None
        if idx >= len(boundaries):
            return None

        # The known resolving vertex itself becomes the next anchor.  Keep
        # its effective sweep coordinate for ordering, but use the actual
        # graph vertex for all distance calculations.
        anchor_rel, anchor, _kind = boundaries[idx]
        idx += 1
        active_boundary_rel = next_boundary(graph, n, rel, anchor_rel, active)




def _frontier_gap_check(graph, dist, k, n, rel, start_vertex, s_t, active, current_full,
                        placements=None):
    """Frontier formulation of the cycle large-gap step.

    From the current resolving boundary, x at forward distance k+1 is the
    first cycle vertex not covered by that boundary.  If the unique distant
    vertex has already appeared on a pendant path or processed cycle segment,
    the next resolver must be chosen in {x,...,x+k}.  Otherwise x may itself
    be the unique distant vertex, so the interval extends once more to
    {x,...,x+k+1}.

    If an ordinary active requirement already offers a position before that
    deadline, it is allowed to steer the sweep and candidate_gap_feasible
    enforces the same maximum jump.  Likewise an already-selected cycle or
    pendant resolver that handles x becomes the next boundary instead of
    creating a redundant group.
    """
    anchor = s_t
    anchor_rel = rel(s_t)

    # Existing selected resolvers ahead of the anchor.  Pendant resolvers use
    # rel(join)+a for ordering but their real graph vertex for distance tests.
    known = []
    for v in current_full:
        if v < n and v != s_t:
            r = rel(v)
            if r > anchor_rel:
                known.append((r, v, 'cycle'))
    for eff, resolver, _join_rel in path_anchor_points(graph, rel, placements, current_full):
        if eff > anchor_rel and resolver != s_t:
            known.append((eff, resolver, 'path'))
    known.append((n, start_vertex, 'start'))
    known.sort(key=lambda z: z[0])
    used = set()

    while True:
        x_rel = anchor_rel + k + 1
        if x_rel >= n:
            return None
        x = (start_vertex + int(x_rel)) % n

        # Existing implementation of the distant-vertex rule: pendant
        # vertices always count, cycle vertices only after the sweep has
        # processed them.
        eligible = eligible_for_distant_check(
            graph, current_full, rel, min(int(anchor_rel), n - 1)
        )
        radius = current_radius(graph, dist, current_full, eligible)
        extension = k + 1 if radius == k + 1 else k
        end_rel = min(x_rel + extension, n)

        # If an already-active condition can place a resolver no later than
        # this deadline, do not add a competing frontier group.  Its actual
        # candidate is still checked by candidate_gap_feasible.
        for g in active:
            if any(anchor_rel < rel(v) <= end_rel
                   for branch in g['candidates'] for v in branch):
                return None

        # Closest already-selected resolver ahead may itself cover x (or leave
        # it as the one allowed distant vertex when radius==k+1).
        next_known = None
        for item in known:
            r, resolver, kind = item
            if r <= anchor_rel or item in used:
                continue
            next_known = item
            break
        if next_known is not None:
            r, resolver, kind = next_known
            if r <= end_rel and dist[resolver][x] <= radius:
                if kind == 'start':
                    return None
                used.add(next_known)
                anchor_rel = r
                anchor = resolver
                continue

        # New frontier requirement.  It is an existence condition: one
        # resolver anywhere in the interval satisfies this step outright.
        last_real = min(int(end_rel), n - 1)
        first_real = int(x_rel)
        if first_real > last_real:
            return None
        cand = {(start_vertex + pos) % n
                for pos in range(first_real, last_real + 1)}
        return {
            'candidates': [cand],
            'kind': 'existence',
            'source': ('frontier', anchor, x),
        }


def large_gap_check(graph, dist, k, n, rel, start_vertex, s_t, active, current_full,
                    placements=None, use_frontier=False):
    if use_frontier:
        return _frontier_gap_check(
            graph, dist, k, n, rel, start_vertex, s_t, active, current_full,
            placements=placements,
        )
    return _legacy_large_gap_check(
        graph, dist, k, n, rel, start_vertex, s_t, active, current_full,
        placements=placements,
    )



def candidate_gap_feasible(graph, dist, rel, previous_vertex, candidate, current_full,
                           placements=None):
    """Check the large-gap condition *before* scoring a candidate.

    The candidate is only feasible if every segment between consecutive
    already-selected cycle resolving vertices from ``previous_vertex`` up to
    ``candidate`` stays within the currently allowed gap size.  Pendant-path
    resolving vertices determine whether the one exceptional distant vertex
    has already been used, while pre-placed cycle resolving vertices lying
    between the two positions split the arc into smaller segments.

    When a distant vertex already exists, at most 2k vertices may lie
    strictly between consecutive resolving vertices; otherwise the bound is
    2k+1.
    """
    n, k = graph.n, graph.k
    prev_rel = rel(previous_vertex)
    cand_rel = rel(candidate)

    # A directional sweep may not move backwards.
    if cand_rel <= prev_rel:
        return False

    eligible = eligible_for_distant_check(graph, current_full, rel, prev_rel)
    radius = current_radius(graph, dist, current_full, eligible)
    max_gap = 2 * k if radius == k else 2 * k + 1

    # Already-selected resolving vertices split the arc into separate
    # gaps.  Cycle vertices use their literal position; pendant resolvers
    # use rel(join)+a so a raw join position cannot incorrectly beat a
    # genuinely closer cycle/wraparound resolver.
    intermediate = {
        rel(v) for v in current_full
        if v < n and prev_rel < rel(v) < cand_rel
    }
    for eff, resolver, _join_rel in path_anchor_points(
            graph, rel, placements, current_full):
        if prev_rel < eff < cand_rel:
            intermediate.add(eff)
    boundaries = [prev_rel] + sorted(intermediate) + [cand_rel]

    return all(
        boundaries[i + 1] - boundaries[i] - 1 <= max_gap
        for i in range(len(boundaries) - 1)
    )

def gap_after(graph, n, rel, s_t_rel, active, S_P_global):
    """Vertices strictly between s_t and the nearest following boundary,
    in sweep order; returns (gap_start_rel, gap_end_rel_exclusive) using
    n as the sentinel for "wraps to the start". The boundary can come
    from an active group's vertex OR an already-placed cycle vertex in
    S_P_global (e.g. a pre-placed mandatory/collapsed vertex) -- ignoring
    the latter would treat a stretch as an unresolved gap even when a
    known resolving vertex already sits inside it providing coverage."""
    next_rel = n
    for g in active:
        for cand_set in g['candidates']:
            for v in cand_set:
                r = rel(v)
                if r > s_t_rel:
                    next_rel = min(next_rel, r)
    for v in S_P_global:
        if v >= n:
            continue  # path vertex, not a cycle position
        r = rel(v)
        if r > s_t_rel:
            next_rel = min(next_rel, r)
    return s_t_rel + 1, next_rel
    """Vertices strictly between s_t and the nearest following active
    vertex, in sweep order; returns (gap_start_rel, gap_end_rel_exclusive)
    using n as the sentinel for "wraps to the start"."""
    next_rel = n
    for g in active:
        for cand_set in g['candidates']:
            for v in cand_set:
                r = rel(v)
                if r > s_t_rel:
                    next_rel = min(next_rel, r)
    return s_t_rel + 1, next_rel




# =====================================================================
# Clean frontier-based sweep (experimental replacement for large-gap
# bookkeeping).  Admissible/residual groups remain persistent; frontier
# groups are generated afresh from the actual current resolving set.
# =====================================================================

def _forward_vertices(group, rel, current_rel):
    """Union of candidate vertices that are not behind current sweep position."""
    return {
        v for branch in group['candidates'] for v in branch
        if rel(v) >= current_rel
    }


def _prune_generated_group_forward(group, rel, current_rel):
    """Prune a newly generated residual to the unprocessed forward region.

    This is applied only to a residual created by the CURRENT placement.
    Untouched active requirements are never silently discarded merely
    because a candidate jumped past them; such a jump is infeasible.
    """
    new_branches = []
    for branch in group['candidates']:
        fwd = {v for v in branch if rel(v) >= current_rel}
        if fwd:
            new_branches.append(fwd)
    if not new_branches:
        return None
    out = dict(group)
    out['candidates'] = new_branches
    return out


def update_active_directional(graph, dist, active, s_t, radius, rel, selected_resolvers=None):
    """Update persistent active groups after placing s_t.

    Groups hit by s_t may discharge or create a residual.  Any portion of
    such a NEW residual behind s_t is removed because that part of the sweep
    is already processed.  A group not hit by s_t is left unchanged; if it
    lies entirely behind s_t the candidate placement is considered infeasible
    by clean_candidate_feasible() before we ever commit to it.
    """
    out = []
    cur_rel = rel(s_t)
    for g in active:
        if not group_contains(g, s_t):
            out.append(g)
            continue
        ng = update_group_after_placement(
            graph, dist, g, s_t, radius, selected_resolvers
        )
        if ng is None:
            continue
        ng = _prune_generated_group_forward(ng, rel, cur_rel)
        if ng is not None:
            out.append(ng)
    return out


def _processed_distant_exists(graph, dist, current_full, rel, current_rel):
    """The one-distant-vertex budget, using the intended sweep convention.

    Pendant vertices always count. Cycle vertices count only in the processed
    part of the cycle.
    """
    eligible = eligible_for_distant_check(graph, current_full, rel, current_rel)
    return distant_vertex_exists(graph, dist, current_full, eligible)


def frontier_group_clean(graph, dist, current_full, rel, current_rel, persistent=None):
    """Generate the temporary frontier interval.

    Scan the unprocessed cycle for vertices not covered (distance > k) by ANY
    currently selected resolver, including pendant-path resolvers.

    If a distant vertex has already occurred on a pendant path or processed
    cycle, the first uncovered vertex must be covered.

    Otherwise the first uncovered vertex may be the unique distant vertex;
    in that case skip it and find the next actually-uncovered vertex.  Existing
    resolvers may already cover one or more intervening positions.

    If y is the uncovered vertex that MUST be covered, the next cycle resolver
    is allowed at y,y+1,...,y+k in forward sweep order.
    """
    n, k = graph.n, graph.k

    uncovered = []
    for pos in range(current_rel + 1, n):
        v = (rel.start_vertex + pos) % n if hasattr(rel, 'start_vertex') else None
        if v is None:
            # rel is a callable; recover the corresponding vertex using the
            # unique cycle vertex whose relative coordinate is pos.
            v = next(u for u in range(n) if rel(u) == pos)
        if all(dist[s][v] > k for s in current_full):
            uncovered.append((pos, v))

    if not uncovered:
        return None

    distant_used = _processed_distant_exists(
        graph, dist, current_full, rel, current_rel
    )

    # Local suppression before the next active requirement:
    # if the still-unprocessed region up to the first feasible position of
    # the next persistent group contains at most one uncovered vertex, and
    # the distant allowance is still unused, that lone vertex may simply be
    # the unique distant vertex.  Do not look deeper into the next active
    # interval to manufacture an additional frontier.
    if not distant_used and persistent:
        next_positions = []
        for idx, g in enumerate(persistent):
            vals = [
                rel(v)
                for branch in g.get('candidates', [])
                for v in branch
                if rel(v) > current_rel
            ]
            if vals:
                next_positions.append((min(vals), idx))

        if next_positions:
            next_req_rel, next_idx = min(next_positions)
            next_group = persistent[next_idx]
            local_uncovered = [
                (pos, v) for pos, v in uncovered
                if pos <= next_req_rel
            ]

            # Only an ordinary admissible interval cuts off the frontier scan
            # this way. Residual / OR / symmetry requirements keep the
            # established frontier behavior.
            if (next_group.get('kind') == 'admissible'
                    and len(local_uncovered) <= 1):
                return None

    if distant_used:
        must_idx = 0
        y_rel, y = uncovered[must_idx]
        end_rel = min(y_rel + k, n - 1)
        cand = {
            u for u in range(n)
            if y_rel <= rel(u) <= end_rel
        }
    else:
        # The first uncovered vertex may remain as the unique distant vertex.
        # If it is the only uncovered vertex left, no further frontier is
        # needed. Otherwise skip it and let the next uncovered vertex define
        # the frontier.
        if len(uncovered) == 1:
            return None

        _, first = uncovered[0]
        y_rel, y = uncovered[1]
        end_rel = min(y_rel + k, n - 1)
        cand = {
            u for u in range(n)
            if y_rel <= rel(u) <= end_rel
        }
    if not cand:
        return None

    return {
        'candidates': [cand],
        'kind': 'frontier',
        'source': ('frontier_clean', y),
        'temporary': True,
    }


def symmetry_group_forward(graph, dist, s_t, current_full, rel):
    """Forward active interval needed to resolve new local symmetry at s_t."""
    others = current_full - {s_t}
    pairs = find_symmetric_pairs(graph, dist, s_t, others)
    if not pairs:
        return None

    k = graph.k
    cur_rel = rel(s_t)
    cand = {
        y for y in range(graph.n)
        if rel(y) > cur_rel
        and all(k_trunc(dist[y][u], k) != k_trunc(dist[y][up], k)
                for (u, up) in pairs)
    }
    return {
        'candidates': [cand],
        'kind': 'existence',
        'source': ('symmetry_clean', tuple(pairs)),
    } if cand else None


def clean_candidate_feasible(graph, dist, persistent, pos, current_full, rel, current_groups=None):
    """Candidate may not jump past an unsatisfied current requirement.

    ``persistent`` is the state that survives between sweep iterations.
    ``current_groups`` may additionally contain temporary requirements such as
    the current frontier.    A temporary frontier is mandatory for the current free arc.  Hitting it
    satisfies that temporary requirement; jumping beyond every frontier
    candidate is infeasible.
    """
    pos_rel = rel(pos)
    radius = current_radius(graph, dist, current_full | {pos})
    blockers = list(current_groups) if current_groups is not None else list(persistent)

    for g in blockers:
        # Temporary frontier is a one-step mandatory requirement.  It is not
        # converted to a persistent residual after a hit; a fresh frontier is
        # recomputed after the placement.
        if g.get('kind') == 'frontier':
            if group_contains(g, pos):
                continue
            if not any(rel(v) >= pos_rel
                       for b in g['candidates'] for v in b):
                return False
            continue

        if group_contains(g, pos):
            # If the hit genuinely discharges the requirement, it is fine.
            # If a residual still exists but lies entirely behind the new
            # sweep position, choosing pos would leave an unresolved
            # requirement in the processed region and is infeasible.
            ng = update_group_after_placement(
                graph, dist, g, pos, radius, current_full | {pos}
            )
            if ng is None:
                continue
            forward_ng = _prune_generated_group_forward(ng, rel, pos_rel)
            if forward_ng is None:
                return False
            if not _forward_vertices(forward_ng, rel, pos_rel):
                return False
        else:
            # A requirement not hit by this candidate must still have a
            # position available without backtracking.
            if not any(rel(v) >= pos_rel
                       for b in g['candidates'] for v in b):
                return False
    return True


def _clean_candidate_choices(groups, rel):
    """Candidate points from the earliest consecutive overlap chain.

    For an OR group, all branches are considered as ordinary candidate sets.
    The sweep does not skip the OR to process a later non-overlapping
    constraint first.
    """
    if not groups:
        return []

    ordered = sorted(groups, key=lambda g: group_earliest(g, rel))
    choices = []

    for b0 in ordered[0]['candidates']:
        cur = set(b0)
        if not cur:
            continue
        for pos in cur:
            choices.append((1, pos))

        depth = 1
        for g in ordered[1:]:
            intersections = [cur & b for b in g['candidates'] if cur & b]
            if not intersections:
                break
            cur = set().union(*intersections)
            depth += 1
            for pos in cur:
                choices.append((depth, pos))

    by_pos = {}
    for depth, pos in choices:
        by_pos[pos] = max(depth, by_pos.get(pos, 0))

    return sorted(
        [(depth, pos) for pos, depth in by_pos.items()],
        key=lambda z: (z[0], rel(z[1])),
        reverse=True,
    )

def _clean_future_burden(future_groups, rel, after_rel):
    """Approximate number of immediate future placements required.

    Future active intervals that share a common feasible cycle vertex count as
    one obligation, consistent with the sweep rule that overlapping active
    intervals may be discharged together.
    """
    forward=[]
    for g in future_groups:
        branches=[]
        for b in g['candidates']:
            fb={v for v in b if rel(v) > after_rel}
            if fb:
                branches.append(fb)
        if branches:
            ng=dict(g)
            ng['candidates']=branches
            forward.append(ng)

    if not forward:
        return 0

    choices=_clean_candidate_choices(forward, rel)
    max_depth=max((depth for depth,_ in choices), default=1)
    return len(forward) - max_depth + 1


def _clean_future_state(graph, dist, persistent_before, pos,
                        current_full_before, rel):
    """Immediate clean-sweep state after hypothetically placing ``pos``.

    Returns the persistent groups after the placement, the newly generated
    symmetry group (if any), the freshly recomputed frontier (if any), and the
    combined list of future requirements.  ``persistent_before`` is not
    mutated.
    """
    trial_full = current_full_before | {pos}
    trial_radius = current_radius(graph, dist, trial_full)

    trial_persistent = update_active_directional(
        graph, dist, persistent_before, pos, trial_radius, rel, trial_full
    )

    trial_sym = symmetry_group_forward(
        graph, dist, pos, trial_full, rel
    )

    future_persistent = list(trial_persistent)
    if trial_sym is not None:
        sig = frozenset().union(*trial_sym['candidates'])
        if not any(
            g.get('source', (None,))[0] == 'symmetry_clean'
            and frozenset().union(*g['candidates']) == sig
            for g in future_persistent
        ):
            future_persistent.append(trial_sym)

    trial_frontier = frontier_group_clean(
        graph, dist, trial_full, rel, rel(pos), future_persistent
    )

    future_groups = list(future_persistent)
    if trial_frontier is not None:
        future_groups.append(trial_frontier)

    return future_persistent, trial_sym, trial_frontier, future_groups


def _clean_next_requirement_rel(future_groups, rel, after_rel, n):
    """Deadline for the next resolver in relative sweep coordinates.

    For each still-active group, take the FURTHEST forward candidate at which
    that group can still be satisfied.  The next resolver must be placed no
    later than the earliest of these group deadlines.

    Example: a requirement {2,3} means the next resolver may be placed at 2
    or 3, hence its deadline is 3 -- not 2.
    """
    deadlines = []
    for g in future_groups:
        vals = [
            rel(v)
            for branch in g['candidates']
            for v in branch
            if rel(v) > after_rel
        ]
        if vals:
            deadlines.append(max(vals))
    return min(deadlines) if deadlines else n


def sweep_from_clean_frontier(graph, dist, initial_groups, start_vertex,
                              S_P_global, placements=None,
                              max_steps=None,
                              defer_initial_special=True):
    """Directional sweep using frontier + persistent intervals only.

    Large-gap intervals/checks are NOT used here.  The frontier is recomputed
    from actual coverage after every placement.  New unresolved local
    symmetries become ordinary forward existence intervals.
    """
    n, k = graph.n, graph.k
    rel = rel_factory(n, start_vertex)
    # Attach start for frontier coordinate recovery without changing old API.
    try:
        rel.start_vertex = start_vertex
    except Exception:
        pass

    persistent = [
        {**g, 'candidates': [set(s) for s in g['candidates']]}
        for g in initial_groups
    ]

    # If requested, an OR requirement whose branches straddle the sweep origin
    # may be treated as an initial wraparound requirement. This is the only OR
    # deferral in the simple algorithm.
    if defer_initial_special:
        for g in persistent:
            if len(g.get('candidates', [])) > 1:
                relvals = [
                    [rel(v) for v in branch]
                    for branch in g['candidates']
                    if branch
                ]
                # Initial OR associated with the starting region: at least one
                # branch is encountered immediately after the start and another
                # branch lies near the end of the sweep order.
                if relvals:
                    mins=[min(vals) for vals in relvals]
                    maxs=[max(vals) for vals in relvals]
                    if min(mins) <= graph.k + 1 and max(maxs) >= graph.n - (graph.k + 1):
                        g['initial_wraparound'] = True

    S_C = []
    # start_vertex may already be in S_P_global (forced/preselected).
    if start_vertex not in S_P_global:
        S_C.append(start_vertex)

    current_anchor = start_vertex
    current_rel = rel(current_anchor)
    steps = 0
    max_steps = max_steps or (4 * n + 10)

    # The sweep origin is a resolving boundary whether it was newly chosen
    # or already preselected.  In either case it must immediately
    # discharge/update every persistent interval it already satisfies.
    current_full = S_P_global | set(S_C)
    radius = current_radius(graph, dist, current_full)
    persistent = update_active_directional(
        graph, dist, persistent, start_vertex, radius, rel, current_full
    )

    # Only a genuinely new placement can introduce a new local symmetry
    # requirement that was not already accounted for during initial setup.
    if start_vertex not in S_P_global:
        sg = symmetry_group_forward(
            graph, dist, start_vertex, current_full, rel
        )
        if sg is not None:
            # The starting symmetry may either participate immediately or be
            # retained for wraparound. Both variants are tried by the driver.
            if defer_initial_special:
                sg['initial_wraparound'] = True
            persistent.append(sg)

    while steps < max_steps:
        current_full = S_P_global | set(S_C)

        # Drop only empty groups; untouched groups behind the anchor indicate
        # a bad sweep path and make this run invalid.
        persistent = [g for g in persistent if any(g['candidates'])]

        frontier = frontier_group_clean(
            graph, dist, current_full, rel, current_rel, persistent
        )
        groups = list(persistent)
        if frontier is not None:
            groups.append(frontier)

        if not groups:
            break

        # If a persistent group is entirely behind the current anchor, this
        # sweep start/path cannot satisfy it without backtracking.
        bad = False
        for g in persistent:
            if not any(rel(v) > current_rel
                       for b in g['candidates'] for v in b):
                bad = True
                break
        if bad:
            break

        # Consider only forward candidates.  On the first step after the
        # initial resolver, an initial symmetry requirement is a wraparound
        # obligation: retain it for feasibility/future processing, but do not
        # use it to generate or rank the immediate next placement.
        selection_groups = groups
        if current_anchor == start_vertex:
            selection_groups = [
                g for g in groups
                if not g.get('initial_wraparound', False)
            ]
            # If there is no other current requirement, the wraparound
            # symmetry itself must still drive the sweep.
            if not selection_groups:
                selection_groups = groups

        forward_groups = []
        for g in selection_groups:
            branches = []
            for b in g['candidates']:
                fb = {v for v in b if rel(v) > current_rel}
                if fb:
                    branches.append(fb)
            if branches:
                ng = dict(g)
                ng['candidates'] = branches
                forward_groups.append(ng)

        if not forward_groups:
            break

        choice = None
        best_key = None
        for depth, pos in _clean_candidate_choices(forward_groups, rel):
            if not clean_candidate_feasible(
                graph, dist, persistent, pos, current_full, rel,
                current_groups=groups
            ):
                continue

            trial_full = current_full | {pos}
            trial_radius = current_radius(graph, dist, trial_full)
            trial_persistent = update_active_directional(
                graph, dist, persistent, pos, trial_radius, rel, trial_full
            )

            trial_sym = symmetry_group_forward(
                graph, dist, pos, trial_full, rel
            )
            future_groups = list(trial_persistent)
            if trial_sym is not None:
                future_groups.append(trial_sym)

            trial_frontier = frontier_group_clean(
                graph, dist, trial_full, rel, rel(pos), future_groups
            )
            if trial_frontier is not None:
                future_groups.append(trial_frontier)

            next_deadline = _clean_next_requirement_rel(
                future_groups, rel, rel(pos), graph.n
            )
            # Prefer greater current overlap, then the later next deadline,
            # and finally the furthest current feasible position.
            key = (next_deadline, rel(pos))
            if best_key is None or key > best_key:
                best_key = key
                choice = pos

        if choice is None:
            break

        S_C.append(choice)
        current_anchor = choice
        current_rel = rel(choice)
        current_full = S_P_global | set(S_C)
        radius = current_radius(graph, dist, current_full)

        persistent = update_active_directional(
            graph, dist, persistent, choice, radius, rel, current_full
        )

        # Frontier is temporary and is simply discarded here.  A fresh one
        # will be computed at the next iteration.
        sg = symmetry_group_forward(
            graph, dist, choice, current_full, rel
        )
        if sg is not None:
            # Avoid duplicate identical symmetry groups.
            sig = frozenset().union(*sg['candidates'])
            if not any(
                g.get('source', (None,))[0] == 'symmetry_clean'
                and frozenset().union(*g['candidates']) == sig
                for g in persistent
            ):
                persistent.append(sg)

        steps += 1

    return set(S_C)


def sweep_from(graph, dist, initial_groups, start_vertex, S_P_global,
               placements=None, max_candidates_per_step=64, use_frontier=False):
    n, k = graph.n, graph.k
    rel = rel_factory(n, start_vertex)
    active = [{**g, 'candidates': [set(s) for s in g['candidates']]} for g in initial_groups]
    S_C = []
    step = 0
    dynamic_join_or_added = set()

    while True:
        current_full = S_P_global | set(S_C)
        if step == 0:
            s_t = start_vertex
        else:
            if not active:
                break
            active.sort(key=lambda g: group_earliest(g, rel))
            # build candidate positions: chain of intersections starting
            # from active[0]'s branches
            candidates = []  # list of (position, chain_len)
            for b0 in active[0]['candidates']:
                cur = set(b0)
                chain_idx = 1
                # always offer "furthest feasible vertex in current reach"
                reach_positions = sorted(cur, key=rel)
                if reach_positions:
                    candidates.append((reach_positions[-1], 1))
                while chain_idx < len(active):
                    nxt = active[chain_idx]
                    extended = False
                    for b in nxt['candidates']:
                        inter = cur & b
                        if inter:
                            cur = inter
                            extended = True
                            reach_positions = sorted(cur, key=rel)
                            candidates.append((reach_positions[-1], chain_idx + 1))
                            break
                    if not extended:
                        break
                    chain_idx += 1

            # dedupe candidate positions, evaluate each
            seen_pos = {}
            for pos, depth in candidates:
                if pos not in seen_pos or depth > seen_pos[pos]:
                    seen_pos[pos] = depth
            best = None
            previous_vertex = S_C[-1]
            for pos, depth in list(seen_pos.items())[:max_candidates_per_step]:
                if not candidate_gap_feasible(
                    graph, dist, rel, previous_vertex, pos, current_full,
                    placements=placements
                ):
                    continue
                radius = current_radius(graph, dist, current_full)
                sim_active = simulate_placement(graph, dist, active, pos, radius)
                # feasibility: no residual group lies behind pos in sweep order
                feasible = all(group_has_vertex_at_or_after(g, rel, rel(pos)) for g in sim_active)
                if not feasible:
                    continue
                discharged_count = len(active) - len(sim_active)
                next_pos_val = group_earliest(sim_active[0], rel) if sim_active else n
                key = (discharged_count, next_pos_val)
                if best is None or key > best[0]:
                    best = (key, pos)
            if best is None:
                # no feasible candidate among the furthest-per-level
                # picks -- search inward through active[0]'s own
                # branches for a feasible position instead of blindly
                # using an infeasible one. (This is exactly the situation
                # the furthest vertex looked fine locally but left a
                # residual behind it in the sweep -- a nearer vertex in
                # the same interval can avoid that.)
                for b0 in active[0]['candidates']:
                    for pos in sorted(b0, key=rel, reverse=True):
                        if not candidate_gap_feasible(
                            graph, dist, rel, previous_vertex, pos, current_full,
                            placements=placements
                        ):
                            continue
                        radius = current_radius(graph, dist, current_full)
                        sim_active = simulate_placement(graph, dist, active, pos, radius)
                        feasible = all(group_has_vertex_at_or_after(g, rel, rel(pos)) for g in sim_active)
                        if not feasible:
                            continue
                        discharged_count = len(active) - len(sim_active)
                        next_pos_val = group_earliest(sim_active[0], rel) if sim_active else n
                        key = (discharged_count, next_pos_val)
                        if best is None or key > best[0]:
                            best = (key, pos)
            if best is None:
                # No candidate satisfies both the residual and large-gap
                # feasibility checks. Keep the old last-resort fallback so
                # the caller can detect/repair an invalid sweep result, but
                # do not label it feasible.
                fallback = sorted(active[0]['candidates'][0], key=rel)[-1]
                s_t = fallback
            else:
                s_t = best[1]

        S_C.append(s_t)
        current_full = S_P_global | set(S_C)
        radius = current_radius(graph, dist, current_full)
        active = simulate_placement(graph, dist, active, s_t, radius)

        # -------------------------------------------------------------
        # Dynamic path-state update at a join vertex.
        #
        # A medium path initially containing one genuine path resolving
        # vertex imposes its ordinary one-resolver admissible interval.
        # If the sweep subsequently selects the join vertex u_i itself,
        # that path now has TWO associated resolving vertices.  The
        # remaining symmetry condition therefore changes to the usual
        # I_i^+ OR I_i^- condition.  This constraint does not exist before
        # u_i is selected, so it has to be inserted dynamically.
        # -------------------------------------------------------------
        if placements is not None:
            for j, placement in placements.items():
                u_i = graph.join_vertex(j)
                if u_i != s_t or j in dynamic_join_or_added:
                    continue

                # Exactly one genuine path resolver, with no previous
                # collapse onto the join vertex: this is the medium-path
                # case that changes from one-resolver to two-resolver when
                # u_i is selected by the sweep.
                if placement.get('collapsed') or len(placement.get('indices', [])) != 1:
                    continue
                _, m = graph.pendant_paths[j]
                if m > 2 * k + 1:
                    continue

                dynamic_join_or_added.add(j)

                # I_i^+/I_i^- is a symmetry-breaking requirement for the
                # cycle k-neighbourhood of u_i.  Do not generate it when
                # the currently fixed resolving set already resolves that
                # neighbourhood.  This mirrors the initial-setup rule in
                # build_initial_active_groups().
                neighbourhood = {(u_i + d) % n for d in range(-k, k + 1)}
                seen_vectors = set()
                neighbourhood_resolved = True
                for v in neighbourhood:
                    vec = tuple(k_trunc(dist[r][v], k) for r in current_full)
                    if vec in seen_vectors:
                        neighbourhood_resolved = False
                        break
                    seen_vectors.add(vec)

                if not neighbourhood_resolved:
                    i_plus = {(u_i + d) % n for d in range(1, k + 2)}
                    i_minus = {(u_i - d) % n for d in range(1, k + 2)}
                    active.append({
                        'candidates': [i_plus, i_minus],
                        'kind': 'existence',
                        'source': ('dynamic_join_or', j),
                    })

        # symmetric pair check (skip for s_1)
        if step > 0:
            others = (S_P_global | set(S_C)) - {s_t}
            pairs = find_symmetric_pairs(graph, dist, s_t, others)
            if pairs:
                prev = S_C[-2]
                s_t_rel = rel(s_t)
                prev_rel = rel(prev)
                direction_back = -1 if s_t_rel > prev_rel else 1  # move toward prev
                moved = s_t
                moved_ok = None
                dist_moved = 0
                max_move = abs(s_t_rel - prev_rel)
                while dist_moved < max_move:
                    dist_moved += 1
                    moved = (s_t - dist_moved) % n if direction_back == -1 else (s_t + dist_moved) % n
                    remaining_pairs = [
                        (u, up) for (u, up) in pairs
                        if k_trunc(dist[moved][u], k) == k_trunc(dist[moved][up], k)
                    ]
                    if not remaining_pairs:
                        moved_ok = moved
                        break

                move_adopted = False
                if moved_ok is not None:
                    radius2 = current_radius(graph, dist, S_P_global | set(S_C[:-1]) | {moved_ok})
                    active_after_move = simulate_placement(graph, dist, active, moved_ok, radius2)
                    extra_residuals = len(active_after_move) > len(active)
                    if not extra_residuals:
                        next_orig = group_earliest(active[0], rel) if active else n
                        next_moved = group_earliest(active_after_move[0], rel) if active_after_move else n
                        if next_moved > next_orig:
                            S_C[-1] = moved_ok
                            active = active_after_move
                            move_adopted = True

                if not move_adopted:
                    # original position retained (either because moving is
                    # infeasible, or the tie-break favours the original);
                    # queue an active interval so the remaining symmetric
                    # pair(s) get resolved later in the sweep.
                    sym_group = symmetric_resolution_group(graph, dist, pairs)
                    if sym_group is not None:
                        active.append(sym_group)

        # Multiple-fully-distant-vertices correction.
        # If the latest greedy placement leaves two or more vertices at
        # truncated distance k+1 from every resolver, move that latest
        # cycle resolver backwards toward its predecessor until at most one
        # fully-distant vertex remains.  Use the same residual/deadline
        # safeguards as the symmetric-pair correction.
        current_full = S_P_global | set(S_C)
        distant_now = fully_distant_vertices(graph, dist, current_full)
        if step > 0 and len(distant_now) > 1:
            prev = S_C[-2]
            original = S_C[-1]
            original_rel = rel(original)
            prev_rel = rel(prev)
            direction_back = -1 if original_rel > prev_rel else 1
            max_move = abs(original_rel - prev_rel)

            moved_ok = None
            moved_active = None
            for dist_moved in range(1, max_move + 1):
                moved = ((original - dist_moved) % n
                         if direction_back == -1
                         else (original + dist_moved) % n)
                trial_full = S_P_global | set(S_C[:-1]) | {moved}
                if len(fully_distant_vertices(graph, dist, trial_full)) > 1:
                    continue
                radius2 = current_radius(graph, dist, trial_full)
                trial_active = simulate_placement(graph, dist, active, moved, radius2)
                if len(trial_active) > len(active):
                    continue
                next_orig = group_earliest(active[0], rel) if active else n
                next_moved = group_earliest(trial_active[0], rel) if trial_active else n
                if next_moved < next_orig:
                    continue
                moved_ok = moved
                moved_active = trial_active
                break

            if moved_ok is not None:
                S_C[-1] = moved_ok
                s_t = moved_ok
                active = moved_active
                current_full = S_P_global | set(S_C)

        # Large gap check: walk forward from s_t through consecutive
        # boundary-to-boundary segments (continuing past pre-placed
        # cycle vertices, stopping at the next active-group boundary or
        # the full wraparound). Trigger threshold and construction radius
        # both depend on whether the "one tolerated exceptional distant
        # vertex" budget is already spent -- and that check is restricted
        # to vertices the sweep has actually already processed (path
        # vertices, or cycle vertices at or before the current segment's
        # anchor), so a not-yet-reached cycle vertex that merely looks
        # distant right now doesn't get mistaken for an already-
        # accounted-for one.
        current_full = S_P_global | set(S_C)
        lg = large_gap_check(
            graph, dist, k, n, rel, start_vertex, S_C[-1], active,
            current_full, placements=placements, use_frontier=use_frontier
        )
        if lg is not None:
            active.append(lg)

        step += 1
        if step > 4 * n + 10:  # safety valve against pathological loops
            break

    return set(S_C)
