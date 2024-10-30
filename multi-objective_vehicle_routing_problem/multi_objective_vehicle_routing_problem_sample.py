import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from scipy.spatial.distance import cdist
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize

# Define the depot and customer coordinates
depot = np.array([50, 50])
customers = np.array(
    [
        [60, 60],
        [55, 45],
        [50, 40],
        [45, 60],
        [40, 50],
        [55, 55],
        [65, 45],
        [45, 55],
        [60, 65],
        [50, 45],
    ]
)

num_customers = len(customers)
max_trucks = 5  # Adjust based on constraints

# Calculate pairwise distances including depot
all_points = np.vstack([depot, customers])
distances = cdist(all_points, all_points)


# Custom random sampling to provide initial permutations without duplicates
class CustomRandomSampling(PermutationRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        return [np.random.permutation(num_customers) for _ in range(n_samples)]


# Define the VRP problem with dynamic truck usage and standardized objectives
class VRPProblem(ElementwiseProblem):
    def __init__(self, max_trucks, distances, max_distance=None, **kwargs):
        super().__init__(
            n_var=num_customers,
            n_obj=2,
            n_constr=0,
            xl=1,
            xu=num_customers,
            **kwargs,
        )
        self.max_trucks = max_trucks
        self.distances = distances
        self.max_distance = max_distance or np.sum(distances)  # Estimate max distance
        self.max_trucks_used = max_trucks  # Set as max for normalization

    def _evaluate(self, x, out, *args, **kwargs):
        # Sort x to get customer visit order and split into routes
        split_points = np.array_split(np.sort(x.astype(int)), self.max_trucks)
        used_trucks = [route for route in split_points if len(route) > 0]

        # Objective 1: Maximum distance traveled by any truck
        max_route_distance = 0
        for route in used_trucks:
            route_distance = 0
            prev_point = 0  # Start at depot
            for customer in route:
                route_distance += self.distances[prev_point, customer + 1]
                prev_point = customer + 1  # Move to the next customer
            route_distance += self.distances[prev_point, 0]  # Return to depot
            max_route_distance = max(max_route_distance, route_distance)

        # Objective 2: Minimize the number of trucks used
        trucks_used = len(used_trucks)

        # Normalize objectives
        normalized_distance = max_route_distance / self.max_distance
        normalized_trucks_used = trucks_used / self.max_trucks_used

        out["F"] = np.array([normalized_distance, normalized_trucks_used])


# Instantiate the problem
problem = VRPProblem(max_trucks=max_trucks, distances=distances)

# Configure the genetic algorithm
algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=CustomRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
)

# Run the optimization
res = minimize(problem, algorithm, ("n_gen", 100), verbose=True)

# Display the results
print("Best solutions found:")
for f in res.F:
    print("Max Distance (Normalized):", f[0], "Trucks Used (Normalized):", f[1])

# Display route assignments for one solution
best_solution = res.X[0]
customer_order = np.argsort(best_solution)
split_routes = np.array_split(customer_order, max_trucks)

# Display the routes for each truck
for i, route in enumerate(split_routes):
    if len(route) > 0:
        print(f"Truck {i+1} will visit customers in this order:", route + 1)

# Plot the best solution
plt.figure(figsize=(10, 8))
plt.scatter(depot[0], depot[1], color="red", label="Depot", s=100, marker="D")
plt.scatter(customers[:, 0], customers[:, 1], color="blue", label="Customer", s=80)

# Label all points
plt.text(depot[0], depot[1], "D", color="red", fontsize=12, ha="center", va="center")
for i, (x, y) in enumerate(customers):
    plt.text(x, y, str(i + 1), color="blue", fontsize=12, ha="center", va="center")

# Colors for each truck
colors = ["orange", "green", "purple", "cyan", "magenta", "yellow", "brown", "pink"]

# Draw paths for each truck
for i, route in enumerate(split_routes):
    if len(route) == 0:
        continue
    route_points = [0] + [customer + 1 for customer in route] + [0]  # Include depot
    color = colors[i % len(colors)]
    for j in range(len(route_points) - 1):
        start = all_points[route_points[j]]
        end = all_points[route_points[j + 1]]
        plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=2,
            label=f"Truck {i + 1}" if j == 0 else "",
        )

# Add labels and legend
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Optimal Routes for Trucks")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
