import csv
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import os

class SmartCityGraph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, from_node, to_node, distance, congestion, accident_likelihood, road_safety, bidirectional=True):
        if from_node not in self.graph:
            self.graph[from_node] = {}
        self.graph[from_node][to_node] = {
            'distance': distance,
            'congestion': congestion,
            'accident_likelihood': accident_likelihood,
            'road_safety': road_safety
        }
        if bidirectional:
            if to_node not in self.graph:
                self.graph[to_node] = {}
            self.graph[to_node][from_node] = {
                'distance': distance,
                'congestion': congestion,
                'accident_likelihood': accident_likelihood,
                'road_safety': road_safety
            }

    def get_neighbors(self, node):
        return self.graph.get(node, {})


class HillClimbingPathFinder:
    def __init__(self, city_graph, weights, max_restarts=100, verbose=False):
        self.city_graph = city_graph
        self.weights = weights
        self.max_restarts = max_restarts
        self.verbose = verbose

    def calculate_edge_cost(self, edge):
        return (edge['distance'] * self.weights['distance'] +
                edge['congestion'] * self.weights['congestion'] +
                edge['accident_likelihood'] * self.weights['accident'] +
                edge['road_safety'] * self.weights['safety'])

    def find_path(self, start, end):
        best_path = None
        best_cost = float('inf')
        best_cost_details = []

        for restart in range(self.max_restarts):
            current_node = start
            path = [current_node]
            total_cost = 0
            visited = set([current_node])
            cost_details = []

            if self.verbose:
                print(f"Restart {restart+1}/{self.max_restarts}")

            while current_node != end:
                neighbors = self.city_graph.get_neighbors(current_node)
                possible_moves = []

                # For each neighbor, calculate the cost and only consider unvisited nodes
                for neighbor, edge in neighbors.items():
                    if neighbor not in visited:
                        cost = self.calculate_edge_cost(edge)
                        possible_moves.append((neighbor, cost, edge))

                if not possible_moves:
                    if self.verbose:
                        print(f"No unexplored neighbors from {current_node}. Dead end.")
                    break

                # Sort by cost and randomly choose among the best (to avoid local minima)
                possible_moves.sort(key=lambda x: x[1])
                min_cost = possible_moves[0][1]
                candidates = [move for move in possible_moves if move[1] == min_cost]
                chosen_neighbor, chosen_cost, chosen_edge = random.choice(candidates)

                total_cost += chosen_cost
                cost_details.append((current_node, chosen_neighbor, chosen_cost, chosen_edge))
                current_node = chosen_neighbor
                path.append(current_node)
                visited.add(current_node)

            # Update best path if a better one is found
            if current_node == end and total_cost < best_cost:
                best_path = path
                best_cost = total_cost
                best_cost_details = cost_details

        return best_path, best_cost, best_cost_details

    def print_cost_breakdown(self, cost_details):
        breakdown = []
        for i, (from_node, to_node, total_cost, edge) in enumerate(cost_details):
            breakdown.append({
                "Step": i + 1,
                "From": from_node,
                "To": to_node,
                "Distance": edge['distance'],
                "Congestion": edge['congestion'],
                "Accident Likelihood": edge['accident_likelihood'],
                "Road Safety": edge['road_safety'],
                "Step Cost": total_cost
            })
        return breakdown

        # Build the graph for visualisation, including edge weights
    def visualize_path(self, path, scenario, method):
        G = nx.DiGraph()
        for from_node in self.city_graph.graph:
            for to_node, edge_data in self.city_graph.graph[from_node].items():
                G.add_edge(from_node, to_node, weight=self.calculate_edge_cost(edge_data))

        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1500, font_size=12)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        if path:
            # Highlight the chosen path in red
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

        title = f"{scenario} - {method} Path"
        plt.title(title)
        os.makedirs("path_visualizations", exist_ok=True)
        filepath = f"path_visualizations/{scenario.replace(' ', '_')}_{method.replace(' ', '_')}.png"
        plt.savefig(filepath)
        plt.close()
        return filepath


class RandomWalkPathFinder(HillClimbingPathFinder):
    def __init__(self, city_graph, weights, max_steps=1000):
        super().__init__(city_graph, weights)
        self.max_steps = max_steps

    def find_path(self, start, end):
        current_node = start
        path = [current_node]
        total_cost = 0
        cost_details = []
        steps_taken = 0
        visited = set([current_node])

        # Randomly walk through the graph, avoiding revisiting nodes, until max_steps or end is reached
        while current_node != end and steps_taken < self.max_steps:
            neighbors = self.city_graph.get_neighbors(current_node)
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            if not unvisited_neighbors:
                break
            next_node = random.choice(unvisited_neighbors)
            edge = neighbors[next_node]
            cost = self.calculate_edge_cost(edge)
            total_cost += cost
            cost_details.append((current_node, next_node, cost, edge))
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
            steps_taken += 1

        if current_node == end:
            return path, total_cost, cost_details
        else:
            return None, float('inf'), cost_details


def save_all_scenarios_to_csv(all_results, filename="hillclimbing_vs_randomwalk_results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Scenario", "Method", "Step", "From", "To", "Distance", "Congestion",
            "Accident Likelihood", "Road Safety", "Step Cost", "Total Cost"
        ])
        writer.writeheader()
        # Write each scenario's breakdown to CSV for later analysis
        for scenario_result in all_results:
            scenario_name = scenario_result['scenario']
            method = scenario_result.get('method', '')
            total_cost = scenario_result['total_cost']
            breakdown = scenario_result.get('breakdown', [])

            for row in breakdown:
                row["Scenario"] = scenario_name
                row["Method"] = method
                row["Total Cost"] = total_cost
                writer.writerow(row)


if __name__ == "__main__":
    city_graph = SmartCityGraph()
    city_graph.add_edge('A', 'B', 10, 1, 0, 0)
    city_graph.add_edge('A', 'C', 8, 3, 1, 1)
    city_graph.add_edge('B', 'D', 5, 2, 0, 0)
    city_graph.add_edge('C', 'D', 4, 1, 0, 0)
    city_graph.add_edge('D', 'E', 6, 2, 0, 1)
    city_graph.add_edge('E', 'F', 7, 3, 1, 0)
    city_graph.add_edge('C', 'F', 9, 1, 0, 0)
    city_graph.add_edge('F', 'G', 5, 2, 1, 1)

    weight_variations = {
        'Light Traffic': {
            'distance': 0.5,
            'congestion': 0.5,
            'accident': 1,
            'safety': 1
        },
        'Heavy Congestion': {
            'distance': 0.5,
            'congestion': 2,
            'accident': 1,
            'safety': 1
        },
        'Balanced': {
            'distance': 0.5,
            'congestion': 1,
            'accident': 1,
            'safety': 1
        },
        'High Safety': {
            'distance': 0.5,
            'congestion': 1,
            'accident': 1,
            'safety': 2
        }
    }

    all_results = []
    start_node = 'A'
    end_node = 'D'

    for scenario, weights in weight_variations.items():
        print(f"\n\n### Testing {scenario} Scenario ###")

        pathfinder = HillClimbingPathFinder(city_graph, weights, max_restarts=50, verbose=False)
        hc_path, hc_cost, hc_details = pathfinder.find_path(start_node, end_node)
        hc_time = time.time()
        pathfinder.visualize_path(hc_path, scenario, "Hill Climbing")

        if hc_path:
            breakdown = pathfinder.print_cost_breakdown(hc_details)
        else:
            breakdown = []

        all_results.append({
            "scenario": scenario,
            "method": "Hill Climbing",
            "total_cost": hc_cost,
            "breakdown": breakdown,
            "path": hc_path
        })

        random_finder = RandomWalkPathFinder(city_graph, weights)
        rw_path, rw_cost, rw_details = random_finder.find_path(start_node, end_node)
        random_finder.visualize_path(rw_path, scenario, "Random Walk")

        if rw_path:
            rw_breakdown = []
            for i, (from_node, to_node, cost, edge) in enumerate(rw_details):
                rw_breakdown.append({
                    "Step": i + 1,
                    "From": from_node,
                    "To": to_node,
                    "Distance": edge['distance'],
                    "Congestion": edge['congestion'],
                    "Accident Likelihood": edge['accident_likelihood'],
                    "Road Safety": edge['road_safety'],
                    "Step Cost": cost
                })
        else:
            rw_breakdown = []

        all_results.append({
            "scenario": scenario,
            "method": "Random Walk",
            "total_cost": rw_cost,
            "breakdown": rw_breakdown,
            "path": rw_path
        })

    save_all_scenarios_to_csv(all_results)
    print("\n📁 All results and path visualizations saved.")
