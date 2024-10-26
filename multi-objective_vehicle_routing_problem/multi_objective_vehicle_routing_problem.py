from pymoo.core.problem import ElementwiseProblem

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random


# PARAMETERS ======================================================

# NDP locations
NUMBER_OF_NDP_LOCATION = 5
RANGE_OF_NDP_LOCATION = (0, 100)


# HDP locations
NUMBER_OF_HDP_LOCATION = 5
RANGE_OF_HDP_LOCATION = (0, 100)


# Map
RANGE_OF_MAP = (
    (0, 100),
    (0, 100),
)  # 2D map, ((xlim), (ylim)), for example: ((0, 100), (0, 100))


# Transportation
NUMBER_OF_VEHICLE = 1
LOCATION_OF_DEPOT = (50, 50)  # This problem contains only one depot

# =================================================================


class Coordinates:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = float(latitude)
        self.longitude = float(longitude)

    def __repr__(self):
        return f"Coordinates(latitude={self.latitude}, longitude={self.longitude})"

    def flat_distance_to(self, other):
        coord1 = np.array([self.latitude, self.longitude])
        coord2 = np.array([other.latitude, other.longitude])

        return np.linalg.norm(coord1 - coord2)


class MapGraph:
    def __init__(self):
        # Dictionary to store graph adjacency list
        # The keys are Coordinates objects, and the values are lists of tuples (neighbor, distance)
        self.graph = defaultdict(list)
        self.depot = None

    def add_depot(self, new_depot: Coordinates):
        """
        Add a new depot to the graph.
        """
        self.depot = new_depot
        if new_depot not in self.graph:
            self.graph[new_depot] = []  # Initialize adjacency list for new location

    def add_location(self, location: Coordinates):
        """
        Add a new location (vertex) to the graph.
        """
        if location not in self.graph:
            self.graph[location] = []  # Initialize adjacency list for new location

    def road_exists(self, start: Coordinates, end: Coordinates):
        """
        Check if a road (edge) already exists between two coordinates.
        """
        # Check if 'end' is already a neighbor of 'start'
        for neighbor, _ in self.graph[start]:
            if neighbor == end:
                return True
        return False

    def add_road(self, start: Coordinates, end: Coordinates):
        """
        Add a road (edge) with distance between two coordinates to the graph.
        Avoid adding duplicate roads.
        """
        if start not in self.graph:
            self.add_location(start)
        if end not in self.graph:
            self.add_location(end)

        # Check if the road already exists to avoid duplicates
        if not self.road_exists(start, end) and not self.road_exists(end, start):
            # Calculate the distance between the two points
            distance = start.flat_distance_to(end)

            # Add the road (edge) in both directions (undirected graph)
            self.graph[start].append((end, distance))
            self.graph[end].append((start, distance))

        else:
            raise ValueError(f"Road between {start} and {end} already exists")

    def visualize_locations(self):
        """
        Visualize the graph with locations as points.
        """
        plt.figure(figsize=(8, 6))

        plt.scatter(
            self.depot.longitude,
            self.depot.latitude,
            color="red",
            s=100,
            label=(
                "Location" if plt.gca().get_legend_handles_labels()[1] == [] else ""
            ),
        )
        plt.text(
            self.depot.longitude,
            self.depot.latitude,
            f"({self.depot.latitude:.2f}, {self.depot.longitude:.2f})",
            fontsize=9,
        )

        # Plot each location and the roads between them
        for location, neighbors in self.graph.items():
            if location == self.depot:
                continue

            # Plot the location
            plt.scatter(
                location.longitude,
                location.latitude,
                color="blue",
                s=100,
                label=(
                    "Location" if plt.gca().get_legend_handles_labels()[1] == [] else ""
                ),
            )
            plt.text(
                location.longitude,
                location.latitude,
                f"({location.latitude:.2f}, {location.longitude:.2f})",
                fontsize=9,
            )

        # Adding labels and showing the plot
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Map Graph Visualization")
        plt.legend(loc="upper left")
        plt.xlim(RANGE_OF_MAP[0])
        plt.ylim(RANGE_OF_MAP[1])
        plt.grid(True)
        plt.show()


class MultiObjectiveVehicleRoutingProblem(ElementwiseProblem):
    def __init__(self):

        random.seed(111)

        self.depot = Coordinates(LOCATION_OF_DEPOT[0], LOCATION_OF_DEPOT[1])
        self.ndp_loc_list = []
        self.hdp_loc_list = []

        # Add NDP locations
        for i in range(NUMBER_OF_NDP_LOCATION + 1):
            self.ndp_loc_list.append(
                Coordinates(
                    round(
                        random.uniform(
                            RANGE_OF_NDP_LOCATION[0], RANGE_OF_NDP_LOCATION[1]
                        ),
                        4,
                    ),
                    round(
                        random.uniform(
                            RANGE_OF_NDP_LOCATION[0], RANGE_OF_NDP_LOCATION[1]
                        ),
                        4,
                    ),
                )
            )

        # Add HDP locations
        for i in range(NUMBER_OF_HDP_LOCATION + 1):
            self.hdp_loc_list.append(
                Coordinates(
                    round(
                        random.uniform(
                            RANGE_OF_HDP_LOCATION[0], RANGE_OF_HDP_LOCATION[1]
                        ),
                        4,
                    ),
                    round(
                        random.uniform(
                            RANGE_OF_HDP_LOCATION[0], RANGE_OF_HDP_LOCATION[1]
                        ),
                        4,
                    ),
                )
            )

        self.map_graph = MapGraph()

        self.map_graph.add_depot(self.depot)

        for location in self.ndp_loc_list:
            self.map_graph.add_location(location)

    def visualize_locations(self):
        self.map_graph.visualize_locations()


def main():
    problem = MultiObjectiveVehicleRoutingProblem()

    problem.visualize_locations()


if __name__ == "__main__":
    main()
