from pymoo.core.problem import ElementwiseProblem
from typing import List, Tuple
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random


# PARAMETERS ======================================================

# Coordinates style mappings:
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


# Roads style mappings:
# Solid line:     '-'
# Dashed line:    '--'
# Dotted line:    ':'
# Dash-dot line:  '-.'

# Random
random.seed(50)


# Plot settings
ENABLE_NDP_CUSTOMERS = True
ENABLE_HDP_CUSTOMERS = True
ENABLE_POINT_LABEL = True
FIG_SIZE = (16, 8)

# Depot - This problem contains only one depot
DEPOT_NAME = "Depot"
DEPOT_LOCATION = (100, 50)
DEFAULT_COLOR_OF_DEPOT = "red"
DEFAULT_MARKER_OF_DEPOT = "s"
DEFAULT_MARKER_SIZE_OF_DEPOT = 100


# NDP customers
NDP_CUSTOMER_NAME = "NDP_Customer"
NUMBER_OF_NDP_CUSTOMER = 5
RANGE_OF_NDP_CUSTOMER = (
    (0, 200),
    (0, 100),
)  # ((xlim), (ylim)), for example: ((0, 200), (0, 100))
DEFAULT_COLOR_OF_NDP_CUSTOMER = "blue"
DEFAULT_MARKER_OF_NDP_CUSTOMER = "o"
DEFAULT_MARKER_SIZE_OF_NDP_CUSTOMER = 100


# HDP customers
HDP_CUSTOMER_NAME = "HDP_Customer"
NUMBER_OF_HDP_CUSTOMER = 5
RANGE_OF_HDP_CUSTOMER = (
    (0, 200),
    (0, 100),
)  # ((xlim), (ylim)), for example: ((0, 200), (0, 100))
DEFAULT_COLOR_OF_HDP_CUSTOMER = "green"
DEFAULT_MARKER_OF_HDP_CUSTOMER = "^"
DEFAULT_MARKER_SIZE_OF_HDP_CUSTOMER = 100


# Map
RANGE_OF_MAP = (
    (0, 200),
    (0, 100),
)  # 2D map, ((xlim), (ylim)), for example: ((0, 100), (0, 100))
DEFAULT_COLOR_OF_ROAD = "black"
DEFAULT_WIDTH_OF_ROAD = 2
DEFAULT_STYLE_OF_ROAD = ""

# Transportation
NUMBER_OF_VEHICLE = 1


# =================================================================


class Coordinates:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        name: str = None,
        marker: str = None,
        marker_size: int = 100,
        marker_color: str = None,
    ):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.name = name
        self.marker = marker
        self.marker_size = marker_size
        self.marker_color = marker_color

    def __repr__(self):
        return f"Coordinates(latitude={self.latitude}, longitude={self.longitude})"


class Depot(Coordinates):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        name: str = DEPOT_NAME,
        marker: str = DEFAULT_MARKER_OF_DEPOT,
        marker_size: int = DEFAULT_MARKER_SIZE_OF_DEPOT,
        marker_color: str = DEFAULT_COLOR_OF_DEPOT,
    ):
        super().__init__(
            latitude=latitude,
            longitude=longitude,
            name=name,
            marker=marker,
            marker_size=marker_size,
            marker_color=marker_color,
        )


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
        super().__init__(
            latitude=latitude,
            longitude=longitude,
            name=name,
            marker=marker,
            marker_size=marker_size,
            marker_color=marker_color,
        )


class NDP_Customer(Customer):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        name: str = NDP_CUSTOMER_NAME,
        marker: str = DEFAULT_MARKER_OF_NDP_CUSTOMER,
        marker_size: int = DEFAULT_MARKER_SIZE_OF_NDP_CUSTOMER,
        marker_color: str = DEFAULT_COLOR_OF_NDP_CUSTOMER,
    ):
        super().__init__(
            latitude=latitude,
            longitude=longitude,
            name=name,
            marker=marker,
            marker_size=marker_size,
            marker_color=marker_color,
        )


class HDP_Customer(Customer):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        name: str = HDP_CUSTOMER_NAME,
        marker: str = DEFAULT_MARKER_OF_HDP_CUSTOMER,
        marker_size: int = DEFAULT_MARKER_SIZE_OF_HDP_CUSTOMER,
        marker_color: str = DEFAULT_COLOR_OF_HDP_CUSTOMER,
    ):
        super().__init__(
            latitude=latitude,
            longitude=longitude,
            name=name,
            marker=marker,
            marker_size=marker_size,
            marker_color=marker_color,
        )


class Road:
    def __init__(self, start: Coordinates, end: Coordinates):
        self.start = start
        self.end = end
        self.distance = self.calculate_distance()

    def calculate_distance(self):
        coord1 = np.array([self.start.latitude, self.start.longitude])
        coord2 = np.array([self.start.latitude, self.start.longitude])

        return np.linalg.norm(coord1 - coord2)

    def get_distance(self):
        return self.distance


class MapGraph:
    def __init__(self):
        self.coordinate_list: List[Coordinates] = []
        self.road_list: List[Tuple[Coordinates, Coordinates, float]] = []
        self.graph = defaultdict(list)

    def add_location(self, location: Coordinates):
        if location not in self.coordinate_list:
            self.coordinate_list.append(location)
            self.graph[location] = []

    def road_exists(self, start: Coordinates, end: Coordinates):
        for neighbor, _ in self.graph[start]:
            if neighbor == end:
                return True
        return False

    def add_road(self, start: Coordinates, end: Coordinates):
        # Check wether the coordinates exist in the graph
        if start not in self.graph:
            self.add_location(start)
        if end not in self.graph:
            self.add_location(end)

        if not self.road_exists(start, end) and not self.road_exists(end, start):
            distance = start.flat_distance_to(end)
            self.graph[start].append((end, distance))
            self.graph[end].append((start, distance))
            # Add the road to the roads list
            self.roads.append((start, end, distance))
        else:
            raise ValueError(f"Road between {start} and {end} already exists")

    def remove_road(self, start: Coordinates, end: Coordinates):
        """
        Remove the road (edge) between two coordinates if it exists.
        """
        # Remove end from start's neighbors
        self.graph[start] = [
            (neighbor, dist) for neighbor, dist in self.graph[start] if neighbor != end
        ]

        # Remove start from end's neighbors
        self.graph[end] = [
            (neighbor, dist) for neighbor, dist in self.graph[end] if neighbor != start
        ]

        # Remove the road from the roads list
        self.roads = [
            (s, e, d)
            for s, e, d in self.roads
            if not ((s == start and e == end) or (s == end and e == start))
        ]

    def add_all_roads(self):
        """
        Adds roads between all pairs of points in the graph if a road doesn't already exist.
        """
        locations = list(self.graph.keys())

        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                start, end = locations[i], locations[j]

                # Add a road if one does not already exist
                if not self.road_exists(start, end):
                    self.add_road(start, end)

    def compose_visualization_coordinates(self):
        """
        Visualize the graph with coordinates as points.
        """

        unique_label: list[str] = []

        for coordinate in self.coordinate_list:
            # Plot all coordinates

            if not ENABLE_NDP_CUSTOMERS and coordinate.name == NDP_CUSTOMER_NAME:
                continue

            if not ENABLE_HDP_CUSTOMERS and coordinate.name == HDP_CUSTOMER_NAME:
                continue

            plt.scatter(
                coordinate.latitude,
                coordinate.longitude,
                color=coordinate.marker_color,
                marker=coordinate.marker,
                s=coordinate.marker_size,
                label=coordinate.name if coordinate.name not in unique_label else "",
            )

            if ENABLE_POINT_LABEL:
                plt.text(
                    coordinate.latitude + 1,
                    coordinate.longitude + 1,
                    coordinate.name,
                    fontsize=9,
                )

            unique_label.append(coordinate.name)

    def visualize(self):
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
        plt.title("Map Graph Visualization")
        plt.legend(loc="upper left")
        plt.xlim(RANGE_OF_MAP[0])
        plt.ylim(RANGE_OF_MAP[1])
        plt.grid(True)
        # plt.tight_layout()
        plt.show()


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

        # Add depot
        self.map_graph.add_location(self.depot)

        for customer in self.ndp_customer_list:
            self.map_graph.add_location(customer)

        for customer in self.hdp_customer_list:
            self.map_graph.add_location(customer)

    def compose_roads(self):
        """
        Visualize the roads on a 2D plot using matplotlib.
        """
        plt.figure(figsize=(10, 8))

        # Plot each road
        for start, end, distance in self.map_graph.roads:
            plt.plot(
                [start, end],
                [start.y, end.y],
                "b-",  # Blue color, circle markers, solid line
                markersize=5,
            )

    def visualize(self, enable_coordinates: bool = True, enable_roads: bool = True):
        plt.figure(figsize=FIG_SIZE)

        if not enable_coordinates and not enable_roads:
            raise Warning("Please enable at least one option to visualize")
        if enable_coordinates:
            self.map_graph.compose_visualization_coordinates()
        # if enable_roads:
        #     self.compose_roads()

        self.map_graph.visualize()


def main():
    problem = MultiObjectiveVehicleRoutingProblem()

    problem.visualize()


if __name__ == "__main__":
    main()
