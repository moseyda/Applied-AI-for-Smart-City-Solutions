import random
import math
import csv
import copy
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import pandas as pd

# Define the city graph
city_graph = {
    "A": [{"node": "B", "distance": 2, "congestion": 0.3}, {"node": "C", "distance": 5, "congestion": 0.6}],
    "B": [{"node": "A", "distance": 2, "congestion": 0.3}, {"node": "C", "distance": 2, "congestion": 0.2}, {"node": "D", "distance": 4, "congestion": 0.5}],
    "C": [{"node": "A", "distance": 5, "congestion": 0.6}, {"node": "B", "distance": 2, "congestion": 0.2}, {"node": "D", "distance": 1, "congestion": 0.3}],
    "D": [{"node": "B", "distance": 4, "congestion": 0.5}, {"node": "C", "distance": 1, "congestion": 0.3}]
}

# Weights for cost
weights = {"distance": 0.7, "congestion": 0.3}

class ACO:
    def __init__(self, graph, weights, n_ants=10, n_best=3, n_iterations=100, evaporation=0.3, alpha=1, beta=3, congestion_decay=0.1, random_exploration=0.25, verbose=False):
        self.graph = graph
        self.weights = weights
        self.pheromone = defaultdict(lambda: defaultdict(lambda: 1))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.evaporation = evaporation
        self.alpha = alpha
        self.beta = beta
        self.congestion_decay = congestion_decay
        self.random_exploration = random_exploration
        self.verbose = verbose

    def run(self, start, end):
        best_path = None
        best_cost = float('inf')
        stagnation_counter = 0
        previous_best_cost = None

        for iteration in range(1, self.n_iterations + 1):
            all_paths = self.construct_solutions(start, end)
            self.spread_pheromone(all_paths, self.n_best)
            self.evaporate_pheromone()
            self.update_congestion(all_paths)

            iteration_best = min(all_paths, key=lambda x: x[1])
            iteration_best_path, iteration_best_cost = iteration_best

            if iteration_best_cost < best_cost:
                best_path, best_cost = iteration_best
                stagnation_counter = 0  # reset stagnation if improvement
            else:
                stagnation_counter += 1

            # Print sample ant paths every 10 iterations + pheromone matrix if verbose
            if iteration % 10 == 0:
                print(f"Iteration {iteration} sample ant paths:")
                for i, (path, cost) in enumerate(all_paths[:3]):
                    print(f"  Ant {i+1}: {'->'.join(path)}, Cost: {cost:.4f}")
                if self.verbose:
                    self.print_pheromone()

            # Early stopping if stagnation threshold reached
            if stagnation_counter >= 10:
                if self.verbose:
                    print(f"Stopping early due to stagnation at iteration {iteration}")
                break

        return best_path, best_cost, iteration, stagnation_counter >= 10

    def construct_solutions(self, start, end):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.construct_path(start, end)
            cost = self.calculate_cost(path)
            all_paths.append((path, cost))
        return all_paths

    def construct_path(self, start, end):
        path = [start]
        visited = set([start])
        current = start

        while current != end:
            neighbors = self.graph[current]
            # Filter neighbors not yet visited
            candidates = [edge for edge in neighbors if edge["node"] not in visited]

            if not candidates:
                # Dead end — cannot proceed
                break

            # Exploration with random probability
            if random.random() < self.random_exploration:
                next_node = random.choice(candidates)["node"]
                path.append(next_node)
                visited.add(next_node)
                current = next_node
                continue

            # Calculate move probabilities with pheromone and heuristic
            move_probs = []
            total_prob = 0
            for edge in candidates:
                pheromone = self.pheromone[current][edge["node"]] ** self.alpha
                heuristic = (1 / (edge["distance"] * (1 + edge["congestion"]))) ** self.beta
                prob = pheromone * heuristic
                move_probs.append((edge["node"], prob))
                total_prob += prob

            if total_prob == 0:
                # No attractive moves, pick random to avoid stuck
                next_node = random.choice(candidates)["node"]
            else:
                # Roulette wheel selection
                r = random.uniform(0, total_prob)
                upto = 0
                for node, prob in move_probs:
                    upto += prob
                    if upto >= r:
                        next_node = node
                        break

            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def calculate_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = next(edge for edge in self.graph[u] if edge["node"] == v)
            cost += self.weights["distance"] * edge["distance"] + self.weights["congestion"] * edge["congestion"]
        return cost

    def spread_pheromone(self, all_paths, n_best):
        # Only best n_best ants deposit pheromone
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, cost in sorted_paths[:n_best]:
            pheromone_deposit = 1.0 / cost if cost > 0 else 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.pheromone[u][v] += pheromone_deposit

    def evaporate_pheromone(self):
        pheromone_floor = 0.01  # minimum pheromone level to maintain exploration
        for u in self.pheromone:
            for v in self.pheromone[u]:
                self.pheromone[u][v] *= (1 - self.evaporation)
                if self.pheromone[u][v] < pheromone_floor:
                    self.pheromone[u][v] = pheromone_floor

    def update_congestion(self, all_paths):
        usage_count = defaultdict(int)
        for path, _ in all_paths:
            for i in range(len(path) - 1):
                usage_count[(path[i], path[i + 1])] += 1
        for u in self.graph:
            for edge in self.graph[u]:
                v = edge["node"]
                usage = usage_count.get((u, v), 0)
                edge["congestion"] += self.congestion_decay * usage
                edge["congestion"] = min(edge["congestion"], 1.0)

    def print_pheromone(self):
        print("Pheromone matrix:")
        for u in sorted(self.pheromone):
            for v in sorted(self.pheromone[u]):
                print(f"  {u}->{v}: {self.pheromone[u][v]:.4f}")
        print()

# Worst path simulator using greedy max-cost
def find_worst_path(graph, start, end, weights):
    def calculate_cost(u, v):
        edge = next(edge for edge in graph[u] if edge["node"] == v)
        return weights["distance"] * edge["distance"] + weights["congestion"] * edge["congestion"]

    visited = set()
    path = [start]
    current = start

    while current != end:
        visited.add(current)
        neighbors = [n for n in graph[current] if n["node"] not in visited]
        if not neighbors:
            break
        next_node = max(neighbors, key=lambda e: calculate_cost(current, e["node"]))["node"]
        path.append(next_node)
        current = next_node

    total_cost = sum(calculate_cost(path[i], path[i + 1]) for i in range(len(path) - 1))
    return path, total_cost

# Simulate congestion based on time
def simulate_time_based_congestion(graph, hour):
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        factor = 1.5
    elif 10 <= hour <= 16:
        factor = 1.1
    else:
        factor = 0.8
    for u in graph:
        for edge in graph[u]:
            base = edge.get("base_congestion", edge["congestion"])
            edge["base_congestion"] = base
            edge["congestion"] = max(0.1, base * factor)

if __name__ == "__main__":
    num_runs = 10
    start_node = "A"
    end_node = "D"
    results = []

    for run in range(1, num_runs + 1):
        hour = 6 + run
        fresh_graph = copy.deepcopy(city_graph)
        simulate_time_based_congestion(fresh_graph, hour)

        aco = ACO(fresh_graph, weights,
                  n_ants=30,
                  evaporation=0.3,
                  alpha=1.0,
                  beta=2.0,
                  congestion_decay=0.05,
                  random_exploration=0.25,
                  verbose=True,
                  n_iterations=100)

        aco_path, aco_cost, iterations, converged = aco.run(start_node, end_node)

        worst_path, worst_cost = find_worst_path(fresh_graph, start_node, end_node, weights)

        results.append({
            "Run": run,
            "Hour": hour,
            "ACO Path": "->".join(aco_path) if aco_path else "No path",
            "ACO Cost": aco_cost,
            "ACO Iterations": iterations,
            "ACO Converged": converged,
            "Worst Path": "->".join(worst_path),
            "Worst Cost": worst_cost
        })

        print(f"\nRun {run} (Hour {hour}):")
        print(f"  ACO Best Path: {'->'.join(aco_path)} with cost {aco_cost:.4f}")
        print(f"  Worst Path: {'->'.join(worst_path)} with cost {worst_cost:.4f}")

    # Save results to CSV
    keys = results[0].keys()
    with open("aco_vs_worst_results.csv", "w", newline="") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    # Load CSV and plot results
    df = pd.DataFrame(results)
    plt.figure(figsize=(12, 6))
    plt.plot(df["Run"], df["ACO Cost"], label="ACO Best Path Cost", marker='o')
    plt.plot(df["Run"], df["Worst Cost"], label="Worst Path Cost", marker='x')
    plt.xlabel("Run (Simulated Hour)")
    plt.ylabel("Path Cost")
    plt.title("ACO vs Worst Path Cost Comparison Over Runs")
    plt.legend()
    plt.grid(True)
    plt.show()
