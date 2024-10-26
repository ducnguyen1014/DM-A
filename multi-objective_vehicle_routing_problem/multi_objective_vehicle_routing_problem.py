from pymoo.core.problem import ElementwiseProblem

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random


# PARAMETERS ======================================================

# Marker style mappings:
# '.': 'point'            # Small point marker
# ',': 'pixel'            # Pixel marker
# 'o': 'circle'           # Circle marker
# 'v': 'triangle_down'    # Downward triangle
# '^': 'triangle_up'      # Upward triangle
# '<': 'triangle_left'    # Leftward triangle
# '>': 'triangle_right'   # Rightward triangle
# '1': 'tri_down'         # Downward triangle with base
# '2': 'tri_up'           # Upward triangle with base
# '3': 'tri_left'         # Leftward triangle with base
# '4': 'tri_right'        # Rightward triangle with base
# '8': 'octagon'          # Octagon shape
# 's': 'square'           # Square shape
# 'p': 'pentagon'         # Pentagon shape
# '*': 'star'             # Star shape
# 'h': 'hexagon1'         # Hexagon shape type 1
# 'H': 'hexagon2'         # Hexagon shape type 2
# '+': 'plus'             # Plus sign marker
# 'x': 'x'                # X marker
# 'D': 'diamond'          # Diamond shape
# 'd': 'thin_diamond'     # Thin diamond shape
# '|': 'vline'            # Vertical line marker
# '_': 'hline'            # Horizontal line marker
# 'P': 'plus_filled'      # Filled plus sign marker
# 'X': 'x_filled'         # Filled X marker


# Random
random.seed(50)


# Plot settings
ENABLE_POINT_LABEL = False
FIG_SIZE = (16, 8)

# Depot - This problem contains only one depot
DEPOT_LOCATION = (100, 50)
DEFAULT_COLOR_OF_DEPOT = "red"
DEFAULT_marker_OF_DEPOT = "s"
DEFAULT_marker_SIZE_OF_DEPOT = 100


# NDP customers
NUMBER_OF_NDP_CUSTOMER = 5
RANGE_OF_NDP_CUSTOMER = (
    (0, 200),
    (0, 100),
)  # ((xlim), (ylim)), for example: ((0, 200), (0, 100))
DEFAULT_COLOR_OF_NDP_CUSTOMER = "blue"
DEFAULT_marker_OF_NDP_CUSTOMER = "o"
DEFAULT_marker_SIZE_OF_NDP_CUSTOMER = 100


# HDP customers
NUMBER_OF_HDP_CUSTOMER = 5
RANGE_OF_HDP_CUSTOMER = (
    (0, 200),
    (0, 100),
)  # ((xlim), (ylim)), for example: ((0, 200), (0, 100))
DEFAULT_COLOR_OF_HDP_CUSTOMER = "green"
DEFAULT_marker_OF_HDP_CUSTOMER = "^"
DEFAULT_marker_SIZE_OF_HDP_CUSTOMER = 100


# Map
RANGE_OF_MAP = (
    (0, 200),
    (0, 100),
)  # 2D map, ((xlim), (ylim)), for example: ((0, 100), (0, 100))


# Transportation
NUMBER_OF_VEHICLE = 1


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


class Depot(Coordinates):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        name: str = "Depot",
        marker: str = DEFAULT_marker_OF_DEPOT,
        marker_size: int = DEFAULT_marker_SIZE_OF_DEPOT,
        marker_color: str = DEFAULT_COLOR_OF_DEPOT,
    ):
        super().__init__(latitude=latitude, longitude=longitude)

        self.name = name
        self.marker = marker
        self.marker_size = marker_size
        self.marker_color = marker_color


class Customer(Coordinates):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        name: str,
        marker: str,
        marker_size: int,
        marker_color: str,
    ):
        super().__init__(latitude=latitude, longitude=longitude)
        self.name = name
        self.marker = marker
        self.marker_size = marker_size
        self.marker_color = marker_color


class NDP_Customer(Customer):
    def __init__(self, latitude: float, longitude: float):
        super().__init__(
            latitude=latitude,
            longitude=longitude,
            name="NDP_Customer",
            marker=DEFAULT_marker_OF_NDP_CUSTOMER,
            marker_size=DEFAULT_marker_SIZE_OF_NDP_CUSTOMER,
            marker_color=DEFAULT_COLOR_OF_NDP_CUSTOMER,
        )


class HDP_Customer(Customer):
    def __init__(self, latitude: float, longitude: float):
        super().__init__(
            latitude=latitude,
            longitude=longitude,
            name="HDP_Customer",
            marker=DEFAULT_marker_OF_HDP_CUSTOMER,
            marker_size=DEFAULT_marker_SIZE_OF_HDP_CUSTOMER,
            marker_color=DEFAULT_COLOR_OF_HDP_CUSTOMER,
        )


class MapGraph:
    def __init__(self):
        # Dictionary to store graph adjacency list
        # The keys are Coordinates objects, and the values are lists of tuples (neighbor, distance)
        self.graph = defaultdict(list)
        self.depot: Depot = None
        self.customer_list: list[Customer] = []

    def set_depot(self, new_depot: Depot):
        """
        Set a new depot in the graph, removing any existing depot if present.
        """
        # Remove the old depot if it exists
        if hasattr(self, "depot") and self.depot in self.graph:
            del self.graph[self.depot]

        # Set the new depot
        self.depot = new_depot

        # Add the new depot to the graph if it’s not already there
        if new_depot not in self.graph:
            self.graph[new_depot] = []  # Initialize adjacency list for the new location

    def add_customer(self, customer: Customer):
        """
        Add a new customer to the graph.
        """
        if customer not in self.graph:
            self.graph[customer] = []  # Initialize adjacency list for new location

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


class MultiObjectiveVehicleRoutingProblem(ElementwiseProblem):
    def __init__(self):
        self.depot: Depot = Depot(DEPOT_LOCATION[0], DEPOT_LOCATION[1])
        self.ndp_customer_list: list[NDP_Customer] = []
        self.hdp_customer_list: list[HDP_Customer] = []

        # Add NDP locations
        for i in range(NUMBER_OF_NDP_CUSTOMER):
            self.ndp_customer_list.append(
                NDP_Customer(
                    round(
                        random.uniform(
                            RANGE_OF_NDP_CUSTOMER[0][0], RANGE_OF_NDP_CUSTOMER[0][1]
                        ),
                        4,
                    ),
                    round(
                        random.uniform(
                            RANGE_OF_NDP_CUSTOMER[1][0], RANGE_OF_NDP_CUSTOMER[1][1]
                        ),
                        4,
                    ),
                )
            )

        # Add HDP locations
        for i in range(NUMBER_OF_HDP_CUSTOMER):
            self.hdp_customer_list.append(
                HDP_Customer(
                    round(
                        random.uniform(
                            RANGE_OF_HDP_CUSTOMER[0][0], RANGE_OF_HDP_CUSTOMER[0][1]
                        ),
                        4,
                    ),
                    round(
                        random.uniform(
                            RANGE_OF_HDP_CUSTOMER[1][0], RANGE_OF_HDP_CUSTOMER[1][1]
                        ),
                        4,
                    ),
                )
            )

        self.map_graph = MapGraph()

        self.map_graph.set_depot(self.depot)

        for customer in self.ndp_customer_list:
            self.map_graph.add_customer(customer)

        for customer in self.hdp_customer_list:
            self.map_graph.add_customer(customer)

    def visualize_coordinates(self):
        """
        Visualize the graph with coordinates as points.
        """
        plt.figure(figsize=FIG_SIZE)

        # Flag to set labels only once
        depot_label_set = False
        ndp_customer_label_set = False
        hdp_customer_label_set = False

        # Plot depot
        plt.scatter(
            self.depot.latitude,
            self.depot.longitude,
            label=self.depot.name if not depot_label_set else "",
            color=self.depot.marker_color,
            marker=self.depot.marker,
            s=self.depot.marker_size,
        )
        if ENABLE_POINT_LABEL:
            plt.text(
                self.depot.latitude + 1,
                self.depot.longitude + 1,
                self.depot.name,
                fontsize=9,
            )
        depot_label_set = True  # Set flag after plotting depot

        # Plot NDP customers
        for customer in self.ndp_customer_list:
            # Plot the coordinates
            plt.scatter(
                customer.latitude,
                customer.longitude,
                color=customer.marker_color,
                s=customer.marker_size,
                label=customer.name if not ndp_customer_label_set else "",
            )
            if ENABLE_POINT_LABEL:
                plt.text(
                    customer.latitude + 1,
                    customer.longitude + 1,
                    customer.name,
                    fontsize=9,
                )
            ndp_customer_label_set = True  # Set flag after plotting first customer

        # Plot HDP customers
        for customer in self.hdp_customer_list:
            # Plot the coordinates
            plt.scatter(
                customer.latitude,
                customer.longitude,
                color=customer.marker_color,
                s=customer.marker_size,
                label=customer.name if not hdp_customer_label_set else "",
            )
            if ENABLE_POINT_LABEL:
                plt.text(
                    customer.latitude + 1,
                    customer.longitude + 1,
                    customer.name,
                    fontsize=9,
                )
            hdp_customer_label_set = True  # Set flag after plotting first customer

        # Adding labels and showing the plot
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Map Graph Visualization")
        plt.legend(loc="upper left")
        plt.xlim(RANGE_OF_MAP[0])
        plt.ylim(RANGE_OF_MAP[1])
        plt.grid(True)
        # plt.tight_layout()
        plt.show()


def main():
    problem = MultiObjectiveVehicleRoutingProblem()

    problem.visualize_coordinates()


if __name__ == "__main__":
    main()
