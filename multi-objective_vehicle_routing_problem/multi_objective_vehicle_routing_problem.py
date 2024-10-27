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
ENABLE_COORDINATES = True
ENABLE_ROADS = True
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
NDP_CUSTOMER_NAME = "NDP"
NUMBER_OF_NDP_CUSTOMER = 5
RANGE_OF_NDP_CUSTOMER = (
    (0, 200),
    (0, 100),
)  # ((xlim), (ylim)), for example: ((0, 200), (0, 100))
DEFAULT_COLOR_OF_NDP_CUSTOMER = "blue"
DEFAULT_MARKER_OF_NDP_CUSTOMER = "o"
DEFAULT_MARKER_SIZE_OF_NDP_CUSTOMER = 100


# HDP customers
HDP_CUSTOMER_NAME = "HDP"
NUMBER_OF_HDP_CUSTOMER = 5
RANGE_OF_HDP_CUSTOMER = (
    (0, 200),
    (0, 100),
)  # ((xlim), (ylim)), for example: ((0, 200), (0, 100))
DEFAULT_COLOR_OF_HDP_CUSTOMER = "green"
DEFAULT_MARKER_OF_HDP_CUSTOMER = "^"
DEFAULT_MARKER_SIZE_OF_HDP_CUSTOMER = 100


# Road
DEFAULT_COLOR_OF_ROAD = "black"
DEFAULT_WIDTH_OF_ROAD = 1
DEFAULT_STYLE_OF_ROAD = "-"


# Map
RANGE_OF_MAP = (
    (0, 220),
    (-20, 120),
)  # 2D map, ((xlim), (ylim)), for example: ((0, 100), (0, 100))


# Transportation
NUMBER_OF_VEHICLE = 1


# =================================================================


class Coordinates:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        name: str,
        marker: str = None,
        marker_size: int = 100,
        marker_color: str = None,
    ):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.name = name  # unique identifier
        self.marker = marker
        self.marker_size = marker_size
        self.marker_color = marker_color

    def __repr__(self):
        return f"Coordinates(latitude={self.latitude}, longitude={self.longitude})"

    def get_name(self):
        return self.name


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
    def __init__(
        self,
        start: Coordinates,
        end: Coordinates,
        style: str = DEFAULT_STYLE_OF_ROAD,
        width: int = DEFAULT_WIDTH_OF_ROAD,
        color: str = DEFAULT_COLOR_OF_ROAD,
    ):
        self.start = start
        self.end = end
        self.distance = self.calculate_distance()
        self.style = style
        self.width = width
        self.color = color

    def calculate_distance(self):
        coord1 = np.array([self.start.latitude, self.start.longitude])
        coord2 = np.array([self.end.latitude, self.end.longitude])

        return np.linalg.norm(coord1 - coord2)

    def get_distance(self):
        return self.distance


class MapGraph:
    def __init__(self):
        self.coordinate_list: List[Coordinates] = []
        self.road_list: List[Road] = []
        self.graph = defaultdict(list)
        self.unique_label: list[str] = []

    def add_location(self, location: Coordinates):
        if not ENABLE_COORDINATES:
            return

        if location not in self.coordinate_list:
            self.coordinate_list.append(location)
            self.graph[location] = []

    def road_exists(self, new_road: Road):
        if new_road in self.road_list:
            return True
        return False

    def add_road(self, start_name: str, end_name: str):
        """
        Add new road according to coordinates' names
        """

        if not ENABLE_ROADS:
            return

        # Check wether the coordinates exist in the graph
        coordinates_with_start_name = [
            coord for coord in self.coordinate_list if coord.get_name() == start_name
        ]

        coordinates_with_end_name = [
            coord for coord in self.coordinate_list if coord.get_name() == end_name
        ]

        for start in coordinates_with_start_name:
            for end in coordinates_with_end_name:
                new_road = Road(start, end)
                if self.road_exists(new_road):
                    return

                self.road_list.append(new_road)
                self.add_location(start)
                self.add_location(end)

                # Add the road (edge) in both directions (undirected graph)
                self.graph[start].append((end, new_road.get_distance()))
                self.graph[end].append((start, new_road.get_distance()))

    def compose_visualization_coordinates(self):
        """
        Visualize the graph with coordinates as points.
        """

        for coordinate in self.coordinate_list:
            # Plot all coordinates
            label = coordinate.name.rsplit("_", 1)[0]

            plt.scatter(
                coordinate.latitude,
                coordinate.longitude,
                color=coordinate.marker_color,
                marker=coordinate.marker,
                s=coordinate.marker_size,
                label=(label if label not in self.unique_label else ""),
            )

            if ENABLE_POINT_LABEL:
                plt.text(
                    coordinate.latitude + 1,
                    coordinate.longitude + 1,
                    coordinate.name,
                    fontsize=9,
                )

            self.unique_label.append(label)

    def compose_visualization_roads(self):
        """
        Visualize the roads on a 2D plot using matplotlib.
        """

        # Plot each road
        for road in self.road_list:
            plt.plot(
                [road.start.latitude, road.end.latitude],
                [road.start.longitude, road.end.longitude],
                color=road.color,
                linestyle=road.style,
                linewidth=road.width,
                label="Road" if "Road" not in self.unique_label else "",
            )
            self.unique_label.append("Road")

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
                    latitude=round(
                        random.uniform(
                            RANGE_OF_NDP_CUSTOMER[0][0], RANGE_OF_NDP_CUSTOMER[0][1]
                        ),
                        4,
                    ),
                    longitude=round(
                        random.uniform(
                            RANGE_OF_NDP_CUSTOMER[1][0], RANGE_OF_NDP_CUSTOMER[1][1]
                        ),
                        4,
                    ),
                    name=f"{NDP_CUSTOMER_NAME}_{i+1}",
                )
            )

        # Add HDP locations
        for i in range(NUMBER_OF_HDP_CUSTOMER):
            self.hdp_customer_list.append(
                HDP_Customer(
                    latitude=round(
                        random.uniform(
                            RANGE_OF_HDP_CUSTOMER[0][0], RANGE_OF_HDP_CUSTOMER[0][1]
                        ),
                        4,
                    ),
                    longitude=round(
                        random.uniform(
                            RANGE_OF_HDP_CUSTOMER[1][0], RANGE_OF_HDP_CUSTOMER[1][1]
                        ),
                        4,
                    ),
                    name=f"{HDP_CUSTOMER_NAME}_{i+1}",
                )
            )

        self.map_graph = MapGraph()

        # Add depot
        self.map_graph.add_location(self.depot)

        # Add NDP customers
        if ENABLE_NDP_CUSTOMERS:
            for customer in self.ndp_customer_list:
                self.map_graph.add_location(customer)

        # Add HDP customers
        if ENABLE_HDP_CUSTOMERS:
            for customer in self.hdp_customer_list:
                self.map_graph.add_location(customer)

        # Add Depot - NDP Customer roads
        if ENABLE_NDP_CUSTOMERS:
            self.map_graph.add_road("Depot", "NDP_1")
            self.map_graph.add_road("Depot", "NDP_3")

            self.map_graph.add_road("NDP_1", "NDP_2")
            self.map_graph.add_road("NDP_1", "NDP_4")
            self.map_graph.add_road("NDP_1", "NDP_5")

            self.map_graph.add_road("NDP_2", "NDP_5")

            self.map_graph.add_road("NDP_4", "NDP_5")

        # Add Depot - HDP Customer roads
        if ENABLE_HDP_CUSTOMERS:
            self.map_graph.add_road("Depot", "HDP_4")
            self.map_graph.add_road("Depot", "HDP_5")

            self.map_graph.add_road("HDP_1", "HDP_3")
            self.map_graph.add_road("HDP_1", "HDP_4")

            self.map_graph.add_road("HDP_2", "HDP_5")

        # Add NDP Customer - HDP Customer roads
        if ENABLE_NDP_CUSTOMERS and ENABLE_HDP_CUSTOMERS:
            self.map_graph.add_road("NDP_1", "HDP_4")
            self.map_graph.add_road("NDP_1", "HDP_5")

            self.map_graph.add_road("NDP_2", "HDP_2")
            self.map_graph.add_road("NDP_2", "HDP_5")

            self.map_graph.add_road("NDP_3", "HDP_1")
            self.map_graph.add_road("NDP_3", "HDP_2")
            self.map_graph.add_road("NDP_3", "HDP_3")
            self.map_graph.add_road("NDP_3", "HDP_4")
            self.map_graph.add_road("NDP_3", "HDP_5")

            self.map_graph.add_road("NDP_4", "HDP_1")
            self.map_graph.add_road("NDP_4", "HDP_4")

            self.map_graph.add_road("NDP_5", "HDP_2")

    def visualize(self):
        plt.figure(figsize=FIG_SIZE)

        self.map_graph.compose_visualization_coordinates()
        self.map_graph.compose_visualization_roads()
        self.map_graph.visualize()


def main():
    problem = MultiObjectiveVehicleRoutingProblem()

    problem.visualize()


if __name__ == "__main__":
    main()
