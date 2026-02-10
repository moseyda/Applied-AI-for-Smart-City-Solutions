import matplotlib.pyplot as plt
import networkx as nx

def calculate_edge_cost(edge, weights):
    """
    Calculate the weighted cost of an edge.

    Args:
        edge (dict): Edge attributes with keys 'distance', 'congestion', 'accident', 'safety'.
        weights (list): Weights for each attribute.

    Returns:
        float: Calculated cost.
    """
    required_keys = ["distance", "congestion", "accident", "safety"]
    if not all(key in edge for key in required_keys):
        raise ValueError(f"Edge missing required keys: {required_keys}")

    return (
        weights[0] * edge["distance"] +
        weights[1] * edge["congestion"] +
        weights[2] * edge["accident"] +
        weights[3] * edge["safety"]
    )

def create_sample_graph():
    """
    Create a sample graph for testing.

    Returns:
        dict: Sample graph represented as an adjacency list.
    """
    graph = {
        "A": [
            {"node": "B", "distance": 5, "congestion": 2, "accident": 0, "safety": 0},
            {"node": "C", "distance": 3, "congestion": 3, "accident": 1, "safety": 1}
        ],
        "B": [
            {"node": "D", "distance": 4, "congestion": 1, "accident": 0, "safety": 0}
        ],
        "C": [
            {"node": "D", "distance": 2, "congestion": 2, "accident": 1, "safety": 1}
        ],
        "D": []
    }
    return graph

def print_graph(graph, weights=[1, 1, 1, 1]):
    """
    Print the graph's connections and edge costs.

    Args:
        graph (dict): Graph represented as an adjacency list.
        weights (list): Weights for cost calculation.
    """
    for node, neighbors in graph.items():
        print(f"Node {node} connects to:")
        for neighbor in neighbors:
            cost = calculate_edge_cost(neighbor, weights)
            print(f"  - {neighbor['node']} (Cost: {cost})")

def visualize_traffic(graph, path=None, weights=[1, 1, 1, 1], cost=None):
    """
    Visualise the traffic graph and highlight the optimal path if provided.

    Args:
        graph (dict): Graph represented as an adjacency list.
        path (list, optional): Optimal path to highlight.
        weights (list): Weights for cost calculation.
        cost (float, optional): Total cost of the optimal path.
    """
    G = nx.DiGraph()

    # Add nodes and edges
    for node in graph:
        G.add_node(node)
        for neighbor in graph[node]:
            edge_cost = calculate_edge_cost(neighbor, weights)
            G.add_edge(
                node,
                neighbor["node"],
                weight=edge_cost,
                distance=neighbor["distance"],
                congestion=neighbor["congestion"],
                accident=neighbor["accident"],
                safety=neighbor["safety"]
            )

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Highlight optimal path
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            width=3,
            alpha=0.8,
            edge_color="red"
        )

    # Edge labels
    edge_labels = {
        (u, v): f"Cost: {G[u][v]['weight']:.1f}\n"
                f"D:{G[u][v]['distance']} C:{G[u][v]['congestion']}"
        for u, v in G.edges()
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=12)

    # Title
    if path and cost is not None:
        plt.title(f"Optimal Path: {' → '.join(path)}\nTotal Cost: {cost:.2f}", fontsize=14)
    elif path:
        plt.title(f"Optimal Path: {' → '.join(path)}", fontsize=14)
    else:
        plt.title("Road Network", fontsize=14)

    plt.axis("off")
    plt.show()
