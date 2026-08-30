"""
Exact implementation of the directional-sweep algorithm for placing cycle
resolving vertices, given a fixed admissible pendant-path choice S_P.

This follows the mechanics established in conversation with the author,
beyond what was in the written excerpt:

  * Candidate comparison uses ONE-STEP lookahead only (compare where the
    *immediate* next active/residual interval would start after each
    candidate; no deeper simulation of the rest of the sweep).389

  * "Admissible interval already satisfied by another pendant path's
    resolving vertex" does NOT simply drop the constraint. It is treated
    like any other discharge: the discharging vertex's own reach is
    checked against the admissible interval's domain, and any leftover
    (vertices in that domain more than `radius` from the discharging
    vertex) becomes a residual active interval, via the same R_I(.)
    mechanism used for placed cycle vertices.

  * "Large gaps" (checked after every placement of s_t, and once more at
    the end to close the loop back to the sweep's start vertex): if the
    gap between s_t and the next active interval's nearest vertex exceeds
    2k+1 vertices, a new active interval is created:
        { y in gap : radius < d(y, s_t) <= 2*radius + 1 }
    where radius = k+1 UNLESS a "distant vertex" already exists in the
    graph (a vertex at k-truncated distance exactly k+1 from *every*
    currently chosen resolving vertex; there can only ever be one, so
    once one exists, radius must drop to k). This check is evaluated
    fresh, globally, each time a large-gap interval is built.

  * Symmetric-unresolved-pair handling: after placing s_t, if there is a
    pair {u, u'} with d_k(u,s_t) = d_k(u',s_t) <= k and both at d_k = k+1
    from every other currently-chosen resolving vertex, try moving s_t
    one step at a time toward s_{t-1} until the pair(s) are broken. The
    move is accepted only if it introduces no residual interval beyond
    what the original placement would have (checked against every active
    interval the moved point falls into), and only if it does not make
    the next required vertex's placement come sooner in the sweep. This
    exception does not apply to s_1 -- there is no s_0 to move toward.

  * Sweep direction is fixed (increasing vertex index mod n). Only the
    starting vertex is varied, over every vertex of the minimum-
    cardinality admissible interval. The behind-facing-residual exception
    ("processed once the sweep returns to the initial part of the
    cycle") applies only to s_1.

I validate this against brute force on small random instances in
test_exact_sweep.py -- including checking whether the one-step lookahead
assumption ever falls short of the true optimum, which the author flagged
as unconfirmed.
"""

from itertools import product

from tadpole_core import k_trunc


# ---------------------------------------------------------------------
# distant-vertex budget
# ---------------------------------------------------------------------

def distant_vertex_exists(graph, dist, resolving_set, eligible_vertices=None):
    """True if some vertex has k-truncated distance exactly k+1 to every
    vertex currently in `resolving_set`.

    `eligible_vertices`: restricts which vertices can count as "the"
    distant vertex. A vertex only counts if it's either a pendant-path
    vertex, or a cycle vertex in the region the sweep has already passed
    through -- a cycle vertex the sweep hasn't reached yet may happen to
    look distant right now, but that's not a real budget concern, just an
    ordinary outstanding requirement the sweep will address later. If
    None (e.g. pre-sweep, static computation with no sweep position to
    speak of), defaults to all vertices."""
    if not resolving_set:
        return False
    k = graph.k
    candidates = eligible_vertices if eligible_vertices is not None else graph.all_vertices
    for w in candidates:
        if w in resolving_set:
            continue
        if all(dist[s][w] >= k + 1 for s in resolving_set):
            return True
    return False


def current_radius(graph, dist, resolving_set, eligible_vertices=None):
    return (graph.k if distant_vertex_exists(graph, dist, resolving_set, eligible_vertices)
            else graph.k + 1)


# ---------------------------------------------------------------------
# admissible interval construction (with discharge -> residual fix)
# ---------------------------------------------------------------------

def cycle_ball(graph, center, radius):
    n = graph.n
    if radius < 0:
        return set()
    return {(center + d) % n for d in range(-radius, radius + 1)}


def cycle_arc(graph, start, count, direction):
    n = graph.n
    return {(start + direction * d) % n for d in range(count)}


def cyclic_interior(graph, domain):
    """Correctly identify the interior of a (presumed contiguous) arc on
    the cycle, by finding vertices whose cyclic neighbour is missing from
    the domain -- robust to the arc wrapping past vertex 0, unlike a
    numeric sort of vertex ids."""
    n = graph.n
    if len(domain) <= 2 or len(domain) == n:
        return set(domain)
    boundary = {v for v in domain if (v - 1) % n not in domain or (v + 1) % n not in domain}
    interior = domain - boundary
    return interior if interior else set(domain)


def residual_of(graph, dist, domain, ref_vertex, radius):
    """Residual active interval after ``ref_vertex`` partially covers ``domain``.

    Every vertex of ``domain`` matters, including its endpoints.  First find
    the vertices of the original requirement that are still farther than
    ``radius`` from ``ref_vertex``.  The residual candidate set is then the
    intersection of their radius-neighbourhoods on the cycle.

    Returns None when the whole domain is already covered.
    """
    if not domain:
        return None

    leftover = {u for u in domain if dist[ref_vertex][u] > radius}
    if not leftover:
        return None

    candidates = set(range(graph.n))
    for u in leftover:
        candidates &= {y for y in range(graph.n) if dist[y][u] <= radius}
        if not candidates:
            break

    return candidates if candidates else None



def candidates_for_targets(graph, dist, targets, radius):
    """Cycle positions within ``radius`` of every target vertex."""
    targets=set(targets)
    if not targets:
        return None
    candidates=set(range(graph.n))
    for u in targets:
        candidates &= {y for y in range(graph.n) if dist[y][u] <= radius}
        if not candidates:
            break
    return candidates if candidates else None


def cyclic_components(graph, vertices):
    """Split a cycle-vertex set into contiguous components."""
    vertices = set(vertices)
    if not vertices:
        return []
    if len(vertices) == graph.n:
        return [set(vertices)]

    n = graph.n
    missing = next(v for v in range(n) if v not in vertices)
    start = (missing + 1) % n

    comps = []
    cur = set()
    for step in range(n):
        v = (start + step) % n
        if v in vertices:
            cur.add(v)
        elif cur:
            comps.append(cur)
            cur = set()
    if cur:
        comps.append(cur)
    return comps


def admissible_target_residual(graph, dist, domain, path_resolver,
                               primary_discharger, additional_resolvers, radius):
    """Target-carrying residual for an ordinary medium-path admissible interval.

    This mechanism is intentionally limited to the ordinary admissible
    interval of a path with one genuine path resolver.  I+/I-, frontier and
    symmetry groups do not use this rule.

    The primary discharger first determines which vertices of ``domain`` are
    not reached within ``radius``.  However, an ENDPOINT of the admissible
    interval that lies exactly at truncated distance k+1 from BOTH the path
    resolver and the primary discharger is a boundary vertex, not a residual
    target.  Such a boundary vertex is therefore removed before the residual
    candidate interval is formed.

    Other already-selected resolvers may then remove further targets they
    already handle.
    """
    targets={
        u for u in domain
        if dist[primary_discharger][u] > radius
    }

    # Boundary vertices of this cycle interval are the vertices with a cycle
    # neighbour outside the domain.  Only the special k+1/k+1 endpoint case is
    # excluded; the opposite endpoint still remains a target when appropriate.
    n = graph.n
    endpoints = {
        u for u in domain
        if ((u - 1) % n not in domain) or ((u + 1) % n not in domain)
    }
    # The endpoints of the original admissible interval do not belong
    # to the residual requirement.
    targets -= endpoints

    for r in additional_resolvers:
        targets={
            u for u in targets
            if dist[r][u] > radius
        }
        if not targets:
            return None

    if not targets:
        return None

    candidates=candidates_for_targets(graph, dist, targets, radius)
    if not candidates:
        return None

    return {
        'candidates': [candidates],
        'kind': 'admissible_residual',
        'targets': set(targets),
        'radius': radius,
        'source': ('admissible_target_residual',),
    }


def initial_admissible_target_residuals(
    graph, dist, domain, path_resolver,
    primary_discharger, additional_resolvers, radius
):
    """Create one initial residual group per contiguous leftover target component."""
    targets = {
        u for u in domain
        if dist[primary_discharger][u] > radius
    }

    n = graph.n
    endpoints = {
        u for u in domain
        if ((u - 1) % n not in domain) or ((u + 1) % n not in domain)
    }
    # The endpoints of the original admissible interval do not belong
    # to the residual requirement.
    targets -= endpoints

    for r in additional_resolvers:
        targets = {
            u for u in targets
            if dist[r][u] > radius
        }
        if not targets:
            return []

    out = []
    for comp in cyclic_components(graph, targets):
        candidates = candidates_for_targets(graph, dist, comp, radius)
        if not candidates:
            continue
        out.append({
            'candidates': [candidates],
            'kind': 'admissible_residual',
            'targets': set(comp),
            'radius': radius,
            'source': ('admissible_target_residual',),
        })
    return out



def build_initial_active_groups(graph, dist, placements, S_P_global):
    """placements: dict j -> path_placement(...) dict.
    Returns (groups, coverage_arcs):
      groups: list of active-group dicts: {'candidates': [set, ...]}
      coverage_arcs: list of cycle-vertex sets that are already covered
        by an EXISTING resolving vertex's reach, even though no discrete
        placement constraint was generated for them (e.g. a long path's
        a==k boundary case: no I+/I- needed, but the existing near-join
        vertex still reaches k+1-a into the cycle). These aren't active
        groups -- nothing needs to be placed there -- but the static
        neighbouring-gap check needs to know about them as anchors, or
        it would treat that stretch as an uncovered gap it isn't."""
    k = graph.k
    groups = []
    coverage_arcs = []

    def _nominal_side_order(u_i, direction):
        return [
            (u_i + direction * step) % graph.n
            for step in range(1, k + 2)
        ]

    def _covered_prefix_length(ordered, external_pendant):
        """Number of consecutive near-join levels already covered."""
        t = 0
        for v in ordered:
            if any(dist[e][v] <= k for e in external_pendant):
                t += 1
            else:
                break
        return t

    def _shift_order(ordered, direction, amount):
        shifted = list(ordered)
        for _ in range(amount):
            shifted = shifted[1:] + [
                (shifted[-1] + direction) % graph.n
            ]
        return shifted

    def _ip_im_or_data(u_i, other_S_P):
        """Effective symmetric OR windows I_i^+ OR I_i^-.

        Start from the two symmetric nominal k+1 windows.  Determine how many
        consecutive near-join *levels* have already been handled on either
        side by external pendant resolvers.  Because the two sides are
        alternatives of ONE OR constraint, the same shift t is applied to
        both branches, preserving their symmetry about the join.

        Thus if two near-join vertices on I^- are already covered, both I^-
        and I^+ shift outward by two positions.
        """
        external_pendant = {e for e in other_S_P if e >= graph.n}

        plus_nom = _nominal_side_order(u_i, +1)
        minus_nom = _nominal_side_order(u_i, -1)

        t_plus = _covered_prefix_length(plus_nom, external_pendant)
        t_minus = _covered_prefix_length(minus_nom, external_pendant)
        t = max(t_plus, t_minus)

        plus_order = _shift_order(plus_nom, +1, t)
        minus_order = _shift_order(minus_nom, -1, t)

        i_plus, i_minus = set(plus_order), set(minus_order)

        discharger_plus = discharger_minus = None
        plus_first = plus_order[0]
        minus_first = minus_order[0]
        for e in other_S_P:
            if dist[e][plus_first] <= k and discharger_plus is None:
                discharger_plus = e
            if dist[e][minus_first] <= k and discharger_minus is None:
                discharger_minus = e

        return i_plus, i_minus, discharger_plus, discharger_minus

    def _same_side_reaches(u_i, resolver, target, radius):
        """Whether resolver reaches target within radius without using u_i.

        For an AND-side requirement, a resolver on the opposite side of the
        join must not count merely because the shortest route through u_i is
        short.  We therefore require a route of length at most ``radius`` that
        is strictly shorter than every route forced through u_i.
        """
        if resolver == u_i or target == u_i:
            return False
        if dist[resolver][target] > radius:
            return False

        through_join = dist[resolver][u_i] + dist[u_i][target]
        return dist[resolver][target] < through_join


    def _ip_im_and_data(u_i, other_S_P):
        """Effective independent AND windows I_i^+ AND I_i^-.

        The two sides are separate mandatory requirements, so each side shifts
        only by the number of consecutive near-join vertices already handled
        from that same side of the join.
        """
        external_pendant = {e for e in other_S_P if e >= graph.n}

        plus_nom = _nominal_side_order(u_i, +1)
        minus_nom = _nominal_side_order(u_i, -1)

        def covered_prefix_same_side(ordered):
            t = 0
            for v in ordered:
                if any(_same_side_reaches(u_i, e, v, k)
                       for e in external_pendant):
                    t += 1
                else:
                    break
            return t

        t_plus = covered_prefix_same_side(plus_nom)
        t_minus = covered_prefix_same_side(minus_nom)

        plus_order = _shift_order(plus_nom, +1, t_plus)
        minus_order = _shift_order(minus_nom, -1, t_minus)

        i_plus, i_minus = set(plus_order), set(minus_order)

        discharger_plus = discharger_minus = None
        plus_first = plus_order[0]
        minus_first = minus_order[0]
        for e in other_S_P:
            if dist[e][plus_first] <= k and discharger_plus is None:
                discharger_plus = e
            if dist[e][minus_first] <= k and discharger_minus is None:
                discharger_minus = e

        return i_plus, i_minus, discharger_plus, discharger_minus


    def join_k_neighbourhood_resolved(u_i):
        """Whether the currently fixed resolving vertices already resolve
        the cycle k-neighbourhood of the join vertex u_i.

        Dedicated I_i^+/I_i^- constraints are symmetry-breaking constraints
        for this neighbourhood.  If the neighbourhood is already resolved by
        S_P_global (which includes any forced/collapsed join vertices), the
        constraint must not be generated in the first place.
        """
        neighbourhood = cycle_ball(graph, u_i, k)
        if len(neighbourhood) <= 1:
            return True
        if not S_P_global:
            return False
        seen = set()
        for v in neighbourhood:
            vec = tuple(min(dist[s][v], k + 1) for s in S_P_global)
            if vec in seen:
                return False
            seen.add(vec)
        return True

    def add_ip_im_or_constraint(u_i, other_S_P):
        """Long-path rule: one resolver in I_i^+ OR I_i^- is enough."""
        if join_k_neighbourhood_resolved(u_i):
            return

        # Start from the nominal symmetric sides.
        plus_nom = set(_nominal_side_order(u_i, +1))
        minus_nom = set(_nominal_side_order(u_i, -1))

        # Only resolvers external to the pendant path that generated this OR
        # may discharge it.
        external_resolvers = set(other_S_P)

        # OR residualisation is symmetric in the offset from the join.  If the
        # vertex at offset j is already handled on EITHER side, then the
        # symmetric pair at offset j no longer contributes to either residual.
        covered_offsets = set()
        for step in range(1, k + 1):
            u_plus = (u_i + step) % graph.n
            u_minus = (u_i - step) % graph.n
            if any(
                dist[e][u_plus] <= k or dist[e][u_minus] <= k
                for e in external_resolvers
            ):
                covered_offsets.add(step)

        # If every symmetric offset is already handled, the OR condition is
        # fully satisfied and disappears.
        if len(covered_offsets) == k:
            return

        remaining_offsets = [
            step for step in range(1, k + 1)
            if step not in covered_offsets
        ]
        plus_targets = {
            (u_i + step) % graph.n for step in remaining_offsets
        }
        minus_targets = {
            (u_i - step) % graph.n for step in remaining_offsets
        }

        if not covered_offsets:
            plus_branch = set(plus_nom)
            minus_branch = set(minus_nom)
        else:
            plus_branch = candidates_for_targets(
                graph, dist, plus_targets, k
            )
            minus_branch = candidates_for_targets(
                graph, dist, minus_targets, k
            )

        # If the join itself is already selected, it cannot distinguish the
        # symmetric pairs that created this OR and must not be allowed as a
        # residual candidate on either side.
        if u_i in S_P_global:
            plus_branch.discard(u_i)
            minus_branch.discard(u_i)

        branches = []
        if plus_branch:
            branches.append(set(plus_branch))
        if minus_branch:
            branches.append(set(minus_branch))

        # Remove duplicate alternatives that can arise after residualisation.
        unique = []
        seen = set()
        for branch in branches:
            key = frozenset(branch)
            if key and key not in seen:
                seen.add(key)
                unique.append(set(branch))

        if unique:
            groups.append({
                'candidates': unique,
                'kind': 'existence',
                'source': ('residual', 'Ipm_or'),
                'join_vertex': u_i,
            })

    def add_ip_im_and_constraints(u_i, other_S_P):
        """Medium-path special rule: BOTH I_i^+ and I_i^- must be met.

        Each side is represented independently.  If one or more already
        selected EXTERNAL resolvers supply a side, they discharge that side
        collectively: a vertex remains a residual target only when it is
        farther than ``radius`` from every such resolver.

        Unlike the OR case, do NOT suppress this structural AND requirement
        merely because the join's k-neighbourhood is currently distinguished.
        """

        i_plus, i_minus, discharger_plus, discharger_minus = _ip_im_and_data(
            u_i, other_S_P
        )
        radius = current_radius(graph, dist, S_P_global)

        def add_and_side(domain, first_vertex, side_source, direction):
            # The AND side represents the requirement that the first k
            # neighbours of u_i in this direction are resolved. The
            # (k+1)-st vertex belongs to the candidate-placement interval only:
            # a new resolver may sit there and still resolve all k required
            # neighbours, but that last vertex itself need not already be
            # resolved in order to discharge the whole side.
            required = {
                (u_i + direction * step) % graph.n
                for step in range(1, k + 1)
            }

            # Any already selected resolver from another pendant path may
            # discharge an individual required target on this side whenever
            # it lies within distance k of that target. Different targets may
            # therefore be discharged by different resolvers.
            leftover = {
                u for u in required
                if all(dist[e][u] > k for e in other_S_P)
            }

            if not leftover:
                return

            # If none of the required targets has been discharged, retain the
            # original side interval. Otherwise replace it by the residual
            # interval determined by the remaining targets.
            if leftover == required:
                groups.append({
                    'candidates': [domain],
                    'kind': 'existence',
                    'source': (side_source,),
                })
                return

            candidates = candidates_for_targets(
                graph, dist, leftover, k
            )
            if candidates:
                groups.append({
                    'candidates': [candidates],
                    'kind': 'existence',
                    'source': ('residual', side_source),
                })

        plus_first = min(i_plus, key=lambda v: (v - u_i) % graph.n)
        minus_first = min(i_minus, key=lambda v: (u_i - v) % graph.n)
        add_and_side(i_plus, plus_first, 'Iplus', +1)
        add_and_side(i_minus, minus_first, 'Iminus', -1)


    for j, placement in placements.items():
        u_i = graph.join_vertex(j)
        this_path_verts = set(graph.path_vertices[j]) | {u_i}
        other_S_P = S_P_global - this_path_verts
        a = placement['a']
        indices = placement['indices']
        collapsed = placement.get('collapsed', False)

        # Every path's near-join resolving vertex (or u_i itself, when
        # collapsed) reaches exactly ball(u_i, k+1-a) onto the cycle,
        # regardless of whatever else this path also contributes. Large
        # gaps are measured between ALL already-set resolving vertices
        # (or interval endpoints) -- not just between admissible-interval
        # endpoints -- so register this reach uniformly for every path.
        coverage_arcs.append(cycle_ball(graph, u_i, k + 1 - a))

        if collapsed:
            # a=0. u_i is mandatory but is NOT an "admissible interval"
            # offering a choice of position -- it's a hard requirement,
            # conceptually part of S_P rather than something the sweep
            # selects among. It is pre-placed by the caller (see
            # collapsed_vertices below) and excluded from the active-group
            # / starting-interval machinery entirely. Only the genuine
            # remaining choice -- I+/I- -- is added here. u_i must also be
            # excluded from its own I+/I- discharge check: it is trivially
            # within k of both its own neighbours, but that doesn't break
            # the symmetry BETWEEN them, which is the actual requirement.
            # If this is a medium path (no genuine path vertex remains),
            # the join vertex alone leaves BOTH sides symmetric, so both
            # one-sided constraints are required. For a long path whose
            # near-join resolver collapses to u_i, the original long-path
            # rule remains an OR.
            if len(indices) == 0:
                add_ip_im_and_constraints(u_i, other_S_P)
            else:
                add_ip_im_or_constraint(u_i, other_S_P)

        elif len(indices) == 1:
            # SHORT path, genuine single resolving vertex s (not u_i
            # itself). Original Claim: some other resolving vertex must
            # exist within k+1 of s, giving domain = ball(u_i, k+1-a).
            # This applies for every a in [1, k+1] uniformly.
            s_global = graph.path_vertices[j][indices[0]]
            r = k + 1 - a
            domain = cycle_ball(graph, u_i, r)
            # Ordinary admissible-interval coverage is based on true
            # k-neighbourhood reach, not the dynamic k/k+1 gap radius.
            radius = graph.k

            # Ordinary admissible intervals use a target-carrying residual.
            # First find a resolver that satisfies the ORIGINAL admissibility
            # condition for this path.  That resolver creates the initial
            # target set; every other already-selected resolver may then
            # remove targets it already handles.
            discharger = None
            for e in other_S_P:
                if dist[s_global][e] <= k + 1:
                    discharger = e
                    break

            if discharger is None:
                if domain:
                    groups.append({
                        'candidates': [domain],
                        'kind': 'admissible',
                        'domain': set(domain),
                        'path_resolver': s_global,
                        'source': ('admissible',),
                    })
            else:
                additional = set(other_S_P) - {discharger}
                residual_groups = initial_admissible_target_residuals(
                    graph, dist, domain, s_global, discharger,
                    additional, radius
                )
                groups.extend(residual_groups)
            if a == k + 1:
                # r=0, so the single admissible interval is {u_i} and
                # u_i is forced. The path now has two associated resolving
                # vertices: its genuine path resolver and the join vertex.
                # Therefore the remaining symmetry-breaking condition is
                # the same as in the two-resolver case: one of I_i^+ or
                # I_i^- is sufficient (OR).
                add_ip_im_or_constraint(u_i, other_S_P)

        elif len(indices) == 2:
            # LONG path: v2 (near-join) is always backed up within
            # exactly k+1 by its own sibling v1, by construction -- no
            # ball-domain existence check needed at all.
            #
            # I+/I- (the dedicated symmetry-breaking constraint) applies
            # only for a < k strictly. At a == k exactly, u_i's two
            # neighbours are already both at distance k+1 from v2 (the
            # saturation cap) -- that boundary case is covered by general
            # coverage / large-gap checking, not this dedicated mechanism.
            if a < k:
                add_ip_im_or_constraint(u_i, other_S_P)
            # a >= k: neighbours already saturated or fully resolved --
            # the uniform coverage_arc registered above already covers
            # the anchoring need; no additional constraint here.

        else:
            raise ValueError(f"unexpected a={a} > k+1 for path {j}")

    return groups, coverage_arcs


def _arc_bounds(graph, s):
    """Return (start, end) of a presumed-contiguous arc `s`, where start
    is the vertex whose cyclic predecessor is missing from `s` and end is
    the vertex whose successor is missing. Falls back to numeric min/max
    for non-contiguous or whole-cycle sets."""
    n = graph.n
    if not s or len(s) == n:
        return None
    starts = [v for v in s if (v - 1) % n not in s]
    ends = [v for v in s if (v + 1) % n not in s]
    if len(starts) == 1 and len(ends) == 1:
        return starts[0], ends[0]
    return min(s), max(s)


def add_static_neighbouring_gap_constraints(graph, dist, groups, anchor_vertices=(),
                                             anchor_arcs=()):
    """Static "neighbouring admissible interval" gap check (Constraints
    section, not the sweep-time Large gaps paragraph): uses a fixed k+1
    boundary logic, no distant-vertex exception.

    Operates on individual candidate arcs (not merged group footprints,
    which can wrongly fuse two disjoint OR-branch arcs into one shape),
    and explicitly treats overlapping/touching arcs as zero gap rather
    than trusting `(lo - hi - 1) % n` blindly -- that formula silently
    wraps a negative (overlap) gap into a large bogus positive one.

    `anchor_vertices`: pre-placed (forced/collapsed) vertices, included
    here as single-point arcs purely so the gap check still has a
    boundary to measure against near them -- omitting them would make a
    genuine large gap around a pre-placed vertex invisible, since it no
    longer has an admissible-interval arc of its own once pre-placed.

    `anchor_arcs`: cycle-vertex sets already covered by an EXISTING
    resolving vertex's reach even though no discrete placement constraint
    exists for them (e.g. a long path's a==k boundary case) -- also
    included as boundary arcs so a genuine gap isn't missed and an
    already-covered stretch isn't mistaken for one.

    NOTE: a vertex adjacent to a forced singleton is NOT automatically
    safe to exclude from a gap just by being within distance k of it --
    that singleton's two neighbours are only resolved from each other if
    some OTHER vertex actually breaks their symmetry (exactly the same
    condition I+/I- discharge already checks). A blanket "within k of any
    forced singleton" exclusion was tried and is NOT generally sound; it
    can silently under-cover a gap when no such discharging vertex
    exists. Left unimplemented pending a correct way to tie this to the
    per-path I+/I- discharge results rather than guessing structurally."""
    n, k = graph.n, graph.k
    if not groups and not anchor_vertices and not anchor_arcs:
        return groups

    arcs = []
    for v in anchor_vertices:
        arcs.append(({v}, v, v))
    for arc_set in anchor_arcs:
        b = _arc_bounds(graph, arc_set)
        if b is not None:
            arcs.append((arc_set, b[0], b[1]))
    for g in groups:
        for cand in g['candidates']:
            b = _arc_bounds(graph, cand)
            if b is not None:
                arcs.append((cand, b[0], b[1]))

    if len(arcs) == 1:
        # A single anchor arc can't form a pairwise gap with anything, but
        # the rest of the cycle -- its complement -- still needs checking:
        # without this, one lone coverage anchor (or one lone admissible
        # interval) would make the entire remaining cycle invisible to
        # the gap check.
        set_a, start_a, end_a = arcs[0]
        gap_len = n - len(set_a)
        if gap_len > 2 * k + 1:
            gap_vertices = set(range(n)) - set_a
            return groups + [{'candidates': [gap_vertices]}]
        return groups

    if len(arcs) < 1:
        return groups

    order = sorted(range(len(arcs)), key=lambda i: arcs[i][1])
    extra = []
    for a_idx, b_idx in zip(order, order[1:] + order[:1]):
        set_a, _, end_a = arcs[a_idx]
        set_b, start_b, _ = arcs[b_idx]
        if set_a & set_b:
            continue  # overlapping/touching -> no gap
        gap_len = (start_b - end_a - 1) % n
        if gap_len <= 0 or gap_len >= n:
            continue
        if gap_len > 2 * k + 1:
            gap_vertices = {(end_a + 1 + d) % n for d in range(gap_len)}
            if gap_vertices & (set_a | set_b):
                continue  # sanity check: shouldn't overlap either arc
            extra.append({'candidates': [gap_vertices]})
    return groups + extra
