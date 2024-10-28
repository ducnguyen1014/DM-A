import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from scipy.spatial.distance import cdist
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize

text_align = 0.5

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
max_trucks = 3  # Adjust based on constraints

# Calculate pairwise distances including depot
all_points = np.vstack([depot, customers])
print(all_points)

distances = cdist(all_points, all_points)
print(distances)


class CustomRandomSampling(PermutationRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        return [
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int),
            np.array([9, 1, 2, 3, 4, 5, 6, 7, 8, 0], dtype=int),
        ]


class VRPProblem(ElementwiseProblem):
    def __init__(self, max_trucks, distances, **kwargs):
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

    def _evaluate(self, x, out, *args, **kwargs):
        # Decode the permutation to a route assignment
        permutation = np.argsort(x)  # Sort indices to get customer visit order
        # split_points = np.array_split(permutation, self.max_trucks)
        split_points = np.array_split(x.astype(int), self.max_trucks)

        # Objective 1: Maximum distance traveled by any truck
        max_route_distance = 0
        for route in split_points:
            route_distance = 0
            prev_point = 0  # Start at depot
            for customer in route:
                route_distance += self.distances[prev_point, customer + 1]
                prev_point = customer + 1  # Move to the next customer
            route_distance += self.distances[prev_point, 0]  # Return to depot
            max_route_distance = max(max_route_distance, route_distance)

        # Objective 2: Minimize the number of trucks used
        trucks_used = sum([1 for route in split_points if len(route) > 0])

        # Set the objective values
        out["F"] = np.array([max_route_distance, trucks_used])


# Instantiate the problem
problem = VRPProblem(max_trucks=max_trucks, distances=distances, hello="hello world")


# Configure the genetic algorithm
algorithm = NSGA2(
    pop_size=2,
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
    print("Max Distance:", f[0], "Trucks Used:", f[1])

# Display route assignments for one solution
best_solution = res.X[0]

customer_order = np.argsort(best_solution)

# Assuming you are using 2 trucks as per the problem setup
split_routes = np.array_split(customer_order, max_trucks)

# Display the routes for each truck
for i, route in enumerate(split_routes):
    print(f"Truck {i+1} will visit customers in this order:", route + 1)

# Plot the best solution
best_permutation = np.argsort(res.X[0])
split_routes = np.array_split(best_permutation, max_trucks)

# Set up plot
plt.figure(figsize=(10, 8))
plt.scatter(depot[0], depot[1], color="red", label="Depot", s=100, marker="D")
plt.scatter(customers[:, 0], customers[:, 1], color="blue", label="Customer", s=80)

# Label all points
plt.text(
    depot[0] + text_align,
    depot[1] + text_align,
    "D",
    color="red",
    fontsize=12,
    ha="center",
    va="center",
)
for i, (x, y) in enumerate(customers):
    plt.text(
        x + text_align,
        y + text_align,
        str(i + 1),
        color="blue",
        fontsize=12,
        ha="center",
        va="center",
    )

# Colors for each truck
colors = ["orange", "green", "purple", "cyan", "magenta", "yellow", "brown", "pink"]

# Draw paths for each truck
for i, route in enumerate(split_routes):
    route_distance = 0
    route_points = (
        [0] + [customer + 1 for customer in route] + [0]
    )  # Route includes return to depot
    color = colors[i % len(colors)]

    # Plot each segment in the route
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
