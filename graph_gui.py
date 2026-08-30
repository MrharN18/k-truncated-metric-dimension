import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

import itertools
import random

# ============================================================
# PARAMETERS
# ============================================================
c = 10
k = 2

# Set k = 0 to use the ordinary (untruncated) metric dimension.
# For k >= 1, the program uses the k-truncated distance.
TRUNC = k + 1
METRIC_MODE = False
SOLVER_K = k

attachments = {}
selected_node = None

highlighted_node = None

annot = None

mode = "path"   # "path" or "resolve"

# ============================================================
# STATE
# ============================================================
G = None
pos = {}
labels = {}
resolving_set = set()

fig, ax = None, None

# ============================================================
# BUILD GRAPH
# ============================================================
def build_graph():
    global G

    G = nx.Graph()

    G.add_nodes_from(range(c))

    # IMPORTANT FIX: allow single vertex cycle
    if c > 1:
        for i in range(c):
            G.add_edge(i, (i + 1) % c)

    node_id = c

    for attach, paths in attachments.items():
        for length in paths:
            prev = attach
            for _ in range(length):
                G.add_node(node_id)
                G.add_edge(prev, node_id)
                prev = node_id
                node_id += 1


# ============================================================
# LAYOUT
# ============================================================
def compute_layout():
    global pos, labels

    pos.clear()
    labels.clear()

    # ========================================================
    # CYCLE / SINGLE VERTEX
    # ========================================================
    if c == 1:
        pos[0] = (0, 0)
        labels[0] = "C0"
    else:
        for i in range(c):
            angle = 2 * np.pi * i / c
            pos[i] = (np.cos(angle), np.sin(angle))
            labels[i] = f"C{i}"

    # ========================================================
    # PATHS
    # ========================================================
    node_id = c

    for attach, paths in attachments.items():

        # ----------------------------------------------------
        # SUBDIVIDED STAR CASE
        # ----------------------------------------------------
        if c == 1:
            m = len(paths)
            start = -(m - 1) / 2

            for j, length in enumerate(paths):
                sy = (start + j) * 0.8
                # sy = 0

                for i in range(length):
                    pos[node_id] = (
                        (i + 1) * 0.8,
                        sy
                    )
                    labels[node_id] = f"P{attach},{i+1}"
                    node_id += 1
            continue

        # ----------------------------------------------------
        # NORMAL CYCLE CASE
        # ----------------------------------------------------
        base_x, base_y = pos[attach]
        angle = 2 * np.pi * attach / c
        dx, dy = np.cos(angle), np.sin(angle)

        off = 0

        for length in paths:
            sx = -dy * off * 0.25
            sy = dx * off * 0.25
            off += 1

            for i in range(length):
                pos[node_id] = (
                    base_x + (i + 1) * 0.8 * dx + sx,
                    base_y + (i + 1) * 0.8 * dy + sy
                )
                labels[node_id] = f"P{attach},{i+1}"
                node_id += 1



# ============================================================
# METRIC
# ============================================================
def truncated_distance(u, v):
    d = nx.shortest_path_length(G, u, v)
    return d if METRIC_MODE else min(d, TRUNC)

def distance(u,v):
    return nx.shortest_path_length(G, u, v)

def vector_of(v):
    return tuple(truncated_distance(v, r) for r in resolving_set)


def all_vectors():
    return {v: vector_of(v) for v in G.nodes}


# ============================================================
# CONFLICTS
# ============================================================
def find_conflicts(vectors):
    groups = {}
    for v, vec in vectors.items():
        groups.setdefault(vec, []).append(v)
    return [g for g in groups.values() if len(g) > 1]


# ============================================================
# DRAW
# ============================================================
def draw():
    ax.clear()

    vectors = all_vectors()
    conflicts = find_conflicts(vectors)

    conflict_map = {}
    for i, group in enumerate(conflicts):
        for v in group:
            conflict_map[v] = i

    colors = []
    for v in G.nodes:
        if v in resolving_set and not v == highlighted_node:
            colors.append("green")
        elif mode == "path" and v == highlighted_node:
            colors.append("red")
        elif v in conflict_map:
            colors.append(f"C{conflict_map[v] % 10}")
        else:
            colors.append("lightblue")

    nx.draw(G, pos, ax=ax,
            node_color=colors,
            node_size=350,
            edge_color="gray")

    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    metric_label = "ordinary metric dimension" if METRIC_MODE else f"{k}-truncated metric dimension"
    ax.set_title(f"Mode: {mode} | c={c} | {metric_label}")
    ax.axis("off")
    
    ax.set_aspect('equal', adjustable='box')

    fig.canvas.draw_idle()


# ============================================================
# CLICK HANDLER (MODE SWITCHING LOGIC)
# ============================================================
def on_click(event):
    global selected_node
    global highlighted_node

    if event.inaxes != ax:
        return

    x, y = event.xdata, event.ydata

    closest = min(G.nodes,
                  key=lambda v: (pos[v][0]-x)**2 + (pos[v][1]-y)**2)

    # ----------------------------
    # PATH MODE
    # ----------------------------
    if mode == "path":

        if closest < c and selected_node != closest:
            selected_node = closest
            highlighted_node = closest
            print(f"Selected attachment node: {selected_node}")
            print(f"Highlighted attachment node: {highlighted_node}")

            draw()
        else:
            selected_node = None
            highlighted_node = None

            draw()
        return

    # ----------------------------
    # RESOLVING MODE
    # ----------------------------
    if mode == "resolve":

        if closest in resolving_set:
            resolving_set.remove(closest)
        else:
            resolving_set.add(closest)

        draw()


# ============================================================
# ADD PATH
# ============================================================
def add_path(event):
    global attachments

    if selected_node is None:
        return

    length = int(slider_path.val)

    if selected_node not in attachments:
        attachments[selected_node] = []

    attachments[selected_node].append(length)

    rebuild()

# ============================================================
# RESET
# ============================================================
def reset(event):
    global attachments, resolving_set
    attachments = {}
    resolving_set = set()
    rebuild()


# ============================================================
# MIN SET
# ============================================================
def compute_min(event):
    global resolving_set
    global highlighted_node

    if highlighted_node:
        highlighted_node = None

        draw()
        
    nodes = list(G.nodes)

    for r in range(1, len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            seen = {}
            ok = True

            for v in G.nodes:
                vec = tuple(truncated_distance(v, r) for r in subset)
                if vec in seen:
                    ok = False
                    break
                seen[vec] = v

            if ok:
                resolving_set = set(subset)
                draw()
                return


# ============================================================
# REBUILD
# ============================================================
def rebuild():
    global TRUNC, METRIC_MODE, SOLVER_K

    build_graph()

    METRIC_MODE = (k == 0)

    if METRIC_MODE:
        # Any truncation threshold larger than the graph diameter leaves all
        # graph distances unchanged. These values are supplied to the
        # heuristic solver so that it also computes the ordinary metric
        # dimension.
        diameter = nx.diameter(G) if len(G) > 1 else 0
        SOLVER_K = max(1, diameter)
        TRUNC = SOLVER_K + 1
    else:
        SOLVER_K = k
        TRUNC = k + 1

    compute_layout()
    draw()

# ============================================================
# UPDATE
# ============================================================
def update(val):
    global c, k, resolving_set

    c = int(slider_c.val)
    k = int(slider_k.val)

    resolving_set.clear()

    rebuild()

# ============================================================
# HOVER
# ============================================================
def on_hover(event):
    global annot

    if event.inaxes != ax or annot is None:
        return

    x, y = event.xdata, event.ydata

    closest = min(G.nodes,
                  key=lambda v: (pos[v][0]-x)**2 + (pos[v][1]-y)**2)

    dist = (pos[closest][0]-x)**2 + (pos[closest][1]-y)**2

    if dist > 0.05:
        annot.set_visible(False)
        fig.canvas.draw_idle()
        return

    vec = vector_of(closest)

    annot.xy = pos[closest]
    annot.set_text(str(vec))
    annot.set_visible(True)

    fig.canvas.draw_idle()

# ============================================================
# MODE SWITCH BUTTON
# ============================================================
def toggle_mode(event):
    global mode
    global highlighted_node

    mode = "resolve" if mode == "path" else "path"

    highlighted_node = None
    
    draw()

# ============================================================
# RANDOM GRAPH
# ============================================================
def random_graph(event):
    global attachments, resolving_set, highlighted_node, selected_node

    resolving_set.clear()
    attachments = {}

    highlighted_node = None
    selected_node = None

    # Randomly attach paths to cycle vertices
    percentage = random.random()

    join_vertices = random.sample(range(c), 2)

    for v in range(c):
        r = random.random()
        # 40% chance this vertex gets attachments
        if r < 0.3 and r < percentage:
        # if v in join_vertices:
            # num_paths = random.randint(1, 3)
            attachments[v] = [
                random.randint(k + 1, 3*k + 2)
                for _ in range(1)
            ]

    rebuild()

# ============================================================
# GENERALISED TADPOLE SWEEP ALGORITHM
# ============================================================

def compute_exact_cycle_extension(event=None):
    """Run the generalised-tadpole algorithm developed in tadpole_exact.py.

    The pendant paths are read directly from the graph drawn in this GUI.
    The algorithm chooses both the pendant-path resolving vertices and the
    cycle resolving vertices; the user does not need to preselect path
    vertices in resolving mode.
    """
    global resolving_set, highlighted_node

    highlighted_node = None

    if METRIC_MODE:
        print("The generalised-tadpole sweep currently targets k >= 1.")
        return

    if c < 3:
        print("The generalised-tadpole sweep requires a cycle of length at least 3.")
        return

    # The current algorithm assumes at most one pendant path at each cycle
    # vertex.  Keep that restriction explicit rather than silently changing
    # the graph supplied by the GUI.
    if any(len(paths) > 1 for paths in attachments.values()):
        print("The sweep algorithm requires at most one pendant path at each cycle vertex.")
        return

    pendant_paths = [
        (attach, paths[0])
        for attach, paths in sorted(attachments.items())
        if paths
    ]

    if not pendant_paths:
        print("Add at least one pendant path before running the sweep algorithm.")
        return

    # The reduced-graph algorithm currently works with path orders in
    # [k+1, 3k+2].  Longer paths should first be reduced modulo 3k+2 as in
    # the thesis reduction; do not silently alter the displayed graph here.
    bad = [(a, m) for a, m in pendant_paths
           if not (SOLVER_K + 1 <= m <= 3 * SOLVER_K + 2)]
    if bad:
        print("The current sweep implementation expects reduced pendant-path "
              f"lengths in [{SOLVER_K + 1}, {3 * SOLVER_K + 2}]. Invalid: {bad}")
        return

    try:
        from tadpole_core import TadpoleGraph, is_k_truncated_resolving
        from tadpole_exact import solve_exact

        tg = TadpoleGraph(n=c, pendant_paths=pendant_paths, k=SOLVER_K)
        solution, info = solve_exact(
            tg,
            allow_brute_fallback=False,
            verbose=True,
        )

        # TadpoleGraph and build_graph() use the same numbering convention:
        # cycle vertices first, followed by each path in attachment insertion
        # order.  We sorted the attachments above, while the GUI graph was
        # built in dictionary insertion order, so map by semantic path
        # position instead of assuming the numeric path ids coincide.
        gui_path_vertex = {}
        next_id = c
        for attach, paths in attachments.items():
            for length in paths:
                for local_idx in range(length):
                    gui_path_vertex[(attach, local_idx)] = next_id
                    next_id += 1

        solver_to_gui = {i: i for i in range(c)}
        for j, (attach, length) in enumerate(tg.pendant_paths):
            for local_idx, solver_vid in enumerate(tg.path_vertices[j]):
                solver_to_gui[solver_vid] = gui_path_vertex[(attach, local_idx)]

        resolving_set = {solver_to_gui[v] for v in solution}

        print("Sweep algorithm solution:", sorted(resolving_set),
              f"size={len(resolving_set)}")
        print("Terminal gaps:", info.get('gaps'))
        print("Path vertices:", sorted(solver_to_gui[v] for v in info.get('S_P', set())))
        print("Cycle vertices:", sorted(solver_to_gui[v] for v in info.get('S_C', set())))
        print("Valid:", is_k_truncated_resolving(tg, solution))
        draw()

    except Exception as exc:
        print(f"Sweep algorithm failed: {exc}")
        raise


# ============================================================
# INIT
# ============================================================
build_graph()
compute_layout()

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)

# sliders
ax_c = plt.axes([0.2, 0.20, 0.6, 0.02])
ax_k = plt.axes([0.2, 0.15, 0.6, 0.02])
ax_p = plt.axes([0.2, 0.10, 0.6, 0.02])

slider_c = Slider(ax_c, "cycle size", 1, 30, valinit=c, valstep=1)
slider_k = Slider(
    ax_k,
    "k (0 = ordinary)",
    0,
    10,
    valinit=k,
    valstep=1
)
slider_path = Slider(ax_p, "path length", 1, 30, valinit=4, valstep=1)

# buttons
ax_add = plt.axes([0.02, 0.02, 0.15, 0.05])
ax_mode = plt.axes([0.19, 0.02, 0.15, 0.05])
ax_min = plt.axes([0.36, 0.02, 0.15, 0.05])
ax_rand = plt.axes([0.53, 0.02, 0.15, 0.05])
ax_reset = plt.axes([0.70, 0.02, 0.15, 0.05])
ax_min_heuristic = plt.axes([0.87, 0.02, 0.15, 0.05])

btn_add = Button(ax_add, "Add path")
btn_mode = Button(ax_mode, "Toggle mode")
btn_min = Button(ax_min, "Min set")
btn_rand = Button(ax_rand, "Random graph")
btn_reset = Button(ax_reset, "Reset")
btn_min_heuristic = Button(ax_min_heuristic, "Exact cycle ext.")

btn_add.on_clicked(add_path)
btn_mode.on_clicked(toggle_mode)
btn_min.on_clicked(compute_min)
btn_min_heuristic.on_clicked(compute_exact_cycle_extension)
btn_reset.on_clicked(reset)
btn_rand.on_clicked(random_graph)

fig.canvas.mpl_connect("button_press_event", on_click)

slider_c.on_changed(update)
slider_k.on_changed(update)

draw()
plt.show()



