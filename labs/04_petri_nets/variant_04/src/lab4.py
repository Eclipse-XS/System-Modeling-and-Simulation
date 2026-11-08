import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Ellipse

def enabled_transitions(marking):
    p1, p2, p3, p4 = marking
    enabled = []

    if p1 >= 1:
        enabled.append("T1")
    if p2 >= 1:
        enabled.append("T2")
        enabled.append("T3")
    if p3 >= 1 and p4 >= 1:
        enabled.append("T4")

    return enabled

def fire_transition(marking, transition):
    p1, p2, p3, p4 = marking

    if transition == "T1":
        return (p1 - 1, p2 + 1, p3 + 1, p4)
    if transition == "T2":
        return (p1, p2 - 1, p3 + 1, p4)
    if transition == "T3":
        return (p1, p2 - 1, p3, p4 + 1)
    if transition == "T4":
        return (p1 + 1, p2, p3 - 1, p4 - 1)

    raise ValueError(f"Unknown transition: {transition}")

def build_reachability_graph(initial_marking):
    graph = nx.DiGraph()
    visited = {initial_marking: "M0"}
    state_order = [initial_marking]
    queue = deque([initial_marking])

    while queue:
        current = queue.popleft()
        current_name = visited[current]

        graph.add_node(current_name, marking=current)

        for transition in enabled_transitions(current):
            new_marking = fire_transition(current, transition)

            if new_marking not in visited:
                new_name = f"M{len(visited)}"
                visited[new_marking] = new_name
                state_order.append(new_marking)
                queue.append(new_marking)
            else:
                new_name = visited[new_marking]

            graph.add_node(new_name, marking=new_marking)
            graph.add_edge(current_name, new_name, transition=transition)

    return graph, visited, state_order

def build_adjacency_matrix(graph, ordered_names):
    matrix = pd.DataFrame(0, index=ordered_names, columns=ordered_names)
    for u, v in graph.edges():
        matrix.loc[u, v] = 1
    return matrix

def build_transition_matrix(graph, ordered_names):
    matrix = pd.DataFrame("-", index=ordered_names, columns=ordered_names)
    for u, v, data in graph.edges(data=True):
        matrix.loc[u, v] = data["transition"]
    return matrix

def draw_graph(graph):
    pos = {
        "M0": (0, 2.0),
        "M1": (0, 1.0),
        "M2": (-1.6, 0),
        "M3": (1.6, 0),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=16,
        width=1.1,
        edge_color="black",
        connectionstyle="arc3,rad=0.0",
        min_source_margin=10,
        min_target_margin=10
    )

    edge_labels = {(u, v): d["transition"] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=9,
        ax=ax,
        rotate=False
    )

    for node, (x, y) in pos.items():
        ellipse = Ellipse(
            (x, y),
            width=0.95,
            height=0.42,
            facecolor="white",
            edgecolor="black",
            linewidth=1.4
        )
        ax.add_patch(ellipse)

        marking = graph.nodes[node]["marking"]
        label = f"{node}\n{marking}"

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9
        )

    ax.set_title("Граф досяжності мережі Петрі", fontsize=12)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.4, 2.4)
    ax.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    initial_marking = (1, 0, 0, 0)

    graph, visited, state_order = build_reachability_graph(initial_marking)
    ordered_names = [visited[state] for state in state_order]

    print("Список станів:")
    for state in state_order:
        print(f"{visited[state]} = {state}")

    adjacency_matrix = build_adjacency_matrix(graph, ordered_names)
    transition_matrix = build_transition_matrix(graph, ordered_names)

    print("\nМатриця суміжності:")
    print(adjacency_matrix)

    print("\nМатриця переходів:")
    print(transition_matrix)

    draw_graph(graph)

if __name__ == "__main__":
    main()