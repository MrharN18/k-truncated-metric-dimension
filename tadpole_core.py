"""
Core data structures for generalised tadpole graphs and k-truncated
resolving sets.

A generalised tadpole graph = cycle C_n (vertices 0..n-1) with r pendant
paths attached at distinct cycle vertices.

Vertex ids:
    0 .. n-1                -> cycle vertices u_0 .. u_{n-1}
    n .. (all_vertices-1)   -> pendant path vertices

For pendant path j (attached at cycle vertex `attach`, with `length`
vertices), the path vertices are stored in `path_vertices[j]` ordered
from the vertex adjacent to the cycle (local index 0) to the leaf /
free end (local index length-1).
"""

from collections import deque
from itertools import combinations


class TadpoleGraph:
    def __init__(self, n, pendant_paths, k):
        """
        n: cycle length (>=3)
        pendant_paths: list of (attach_vertex, length) pairs, attach_vertex
                       in range(n), distinct across the list
        k: truncation radius for the k-truncated metric dimension
        """
        assert n >= 3
        attach_vertices = [a for a, _ in pendant_paths]
        assert len(set(attach_vertices)) == len(attach_vertices), \
            "pendant paths must attach at distinct cycle vertices"
        self.n = n
        self.k = k
        self.pendant_paths = pendant_paths
        self._build()

    def _build(self):
        n = self.n
        self.adj = {v: set() for v in range(n)}
        for i in range(n):
            self.adj[i].add((i + 1) % n)
            self.adj[(i + 1) % n].add(i)

        self.path_vertices = {}
        next_id = n
        for j, (attach, length) in enumerate(self.pendant_paths):
            verts = []
            prev = attach
            for _ in range(length):
                vid = next_id
                next_id += 1
                self.adj[vid] = set()
                self.adj[prev].add(vid)
                self.adj[vid].add(prev)
                verts.append(vid)
                prev = vid
            self.path_vertices[j] = verts

        self.all_vertices = list(range(next_id))

    def join_vertex(self, j):
        return self.pendant_paths[j][0]

    def path_length(self, j):
        return self.pendant_paths[j][1]

    def bfs_dist(self, src):
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def all_pairs_dist(self):
        return {v: self.bfs_dist(v) for v in self.all_vertices}

    def cycle_dist(self, i, j):
        """Distance between two cycle vertices, measured along the cycle only
        (equals graph distance since paths never shortcut the cycle)."""
        d = abs(i - j) % self.n
        return min(d, self.n - d)


def k_trunc(dist, k):
    return min(dist, k + 1)


def is_k_truncated_resolving(graph, S, dist=None):
    """Check whether S is a k-truncated resolving set of `graph`."""
    if not S:
        return len(graph.all_vertices) <= 1
    if dist is None:
        dist = graph.all_pairs_dist()
    k = graph.k
    seen = {}
    for v in graph.all_vertices:
        vec = tuple(k_trunc(dist[s][v], k) for s in S)
        if vec in seen:
            return False
        seen[vec] = v
    return True


def brute_force_min_resolving(graph, max_size=None, dist=None):
    """Exhaustive search for a minimum k-truncated resolving set.
    Only tractable for small graphs -- intended as a correctness oracle
    and a fallback, not as the primary algorithm."""
    if dist is None:
        dist = graph.all_pairs_dist()
    verts = graph.all_vertices
    max_size = max_size or len(verts)
    for size in range(1, max_size + 1):
        for S in combinations(verts, size):
            if is_k_truncated_resolving(graph, S, dist):
                return set(S)
    return None


def greedy_repair(graph, S, dist=None):
    """If S is not a valid k-truncated resolving set, greedily add vertices
    (each time picking the vertex that eliminates the most remaining
    collisions) until it is. Not guaranteed minimum, but guaranteed to
    terminate with a valid resolving set (adding all vertices always works)."""
    if dist is None:
        dist = graph.all_pairs_dist()
    S = set(S)
    k = graph.k
    while not is_k_truncated_resolving(graph, S, dist):
        # group vertices currently tied
        groups = {}
        for v in graph.all_vertices:
            vec = tuple(k_trunc(dist[s][v], k) for s in S)
            groups.setdefault(vec, []).append(v)
        tied = [grp for grp in groups.values() if len(grp) > 1]
        best_v, best_gain = None, -1
        for cand in graph.all_vertices:
            if cand in S:
                continue
            # gain = number of currently-tied groups that cand splits apart
            gain = 0
            for grp in tied:
                sub = {}
                for v in grp:
                    sub.setdefault(k_trunc(dist[cand][v], k), []).append(v)
                if len(sub) > 1:
                    gain += 1
            if gain > best_gain:
                best_gain, best_v = gain, cand
        S.add(best_v)
    return S
