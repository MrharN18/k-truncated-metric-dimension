"""
Deterministic pendant-path vertex placement, per the author's description:

The search space is exactly r+1 global configurations for r pendant
paths: either every path uses terminal gap k, or exactly one path (try
each of the r) uses terminal gap k+1 while the rest use k.

Given a path's order m and its assigned terminal gap g in {k, k+1}:
  - v1 sits at local index idx1 = m-1-g (0 = nearest the join vertex,
    m-1 = leaf). The terminal gap is the set of vertices from the leaf
    up to (not including) v1.
  - Short paths (m in [k+1, 2k+1]): v1 is the only resolving vertex.
    a = d(v1, u_i) = m - g.
  - Long paths (m in [2k+2, 3k+2]): a second vertex v2 is placed at
    distance k+1 from v1, moving toward the join vertex:
    idx2 = idx1 - (k+1). a = d(v2, u_i) = idx2 + 1.

Collapse: if the computed index is negative, that vertex falls off the
path entirely and lands on u_i itself. This can only happen for the
one path (if any) assigned gap k+1, and only at the minimum order for
its class:
  - short path, m = k+1, g = k+1  -> v1 collapses onto u_i
  - long path,  m = 2k+2, g = k+1 -> v2 collapses onto u_i (v1 remains
    a genuine path vertex)

Special symmetry rule at the join vertex:
  - if the join vertex u_i is the ONLY resolving vertex associated with a
    medium path (the path resolver collapses to u_i), both I_i^+ and I_i^-
    must contain a resolving vertex (AND).
  - if a medium path has one genuine path resolver and a=k+1 forces u_i
    through the singleton admissible interval {u_i}, then there are two
    associated resolving vertices. The remaining condition is I_i^+ OR
    I_i^-: one side is sufficient.
  - for a long path, the existing two-resolving-vertex rule is likewise OR
    whenever its dedicated I_i^+/I_i^- condition applies.
"""


def path_placement(m, k, g):
    """Returns a dict describing the deterministic placement for a path
    of order m under assigned terminal gap g (g in {k, k+1}).

    Fields:
      indices: list of local indices (0, 1, or 2 entries) of genuine
               path vertices to add to S_P.
      a: distance from the near-join resolving "vertex" to u_i (0 if
         collapsed onto u_i).
      collapsed: True if the near-join vertex is u_i itself rather than
                 a genuine path vertex.
    """
    assert k + 1 <= m <= 3 * k + 2
    idx1 = m - 1 - g

    if m <= 2 * k + 1:
        # short path: one resolving vertex
        if idx1 < 0:
            return {'indices': [], 'a': 0, 'collapsed': True}
        a1 = idx1 + 1
        return {'indices': [idx1], 'a': a1, 'collapsed': False}
    else:
        # long path: two resolving vertices
        if idx1 < 0:
            raise ValueError(f"unexpected collapse of v1 for long path m={m}, g={g}")
        idx2 = idx1 - (k + 1)
        if idx2 < 0:
            # v2 collapses onto u_i; v1 remains genuine
            return {'indices': [idx1], 'a': 0, 'collapsed': True}
        a2 = idx2 + 1
        return {'indices': [idx2, idx1], 'a': a2, 'collapsed': False}


def global_configurations(path_orders, k):
    """path_orders: list of path orders (one per path, in path index
    order). Returns a list of configurations, each a list of assigned
    gaps (one per path) -- exactly r+1 configurations: all-k, or exactly
    one path bumped to k+1."""
    r = len(path_orders)
    configs = [[k] * r]
    for i in range(r):
        cfg = [k] * r
        cfg[i] = k + 1
        configs.append(cfg)
    return configs
