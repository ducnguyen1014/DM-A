from pymoo.operators.sampling.rnd import Sampling
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.mutation import Mutation
from pymoo.optimize import minimize
from pymoo.core.result import Result

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from itertools import product
from collections import Counter

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
SEED = 30
random.seed(SEED)
np.random.seed(SEED)

ENABLE_IND_HDP_PROBLEM = False
ENABLE_DEP_HDP_PROBLEM_2OBJECTIVE = False
ENABLE_DEP_HDP_PROBLEM_3OBJECTIVE = False
ENABLE_DEP_HDP_PROBLEM_2OBJECTIVE_MEMETIC = False

# Plot settings
ENABLE_COORDINATES = True
ENABLE_ROADS = True

ENABLE_GRID = True
ENABLE_POINT_LABEL = True
FIG_SIZE = (16, 8)

ENABLE_SOLUTION_VISUALIZATION = True


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
NUMBER_OF_HDP_CUSTOMER = 10
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
MAX_NUMBER_OF_TRUCK = 10


# Coefficients
WEIGHT_OF_MAX_NUMBER_OF_TRUCK_ = 0.5  # Range: 0.0 - 1.0
WEIGHT_OF_MAX_DISTANCE_AMONG_TRUCKS = 0.5  # Range: 0.0 - 1.0


# Route colors
ROUTE_COLORS = [
    "#FF5733",  # Vivid Orange
    "#33A1FD",  # Sky Blue
    "#8DFF33",  # Lime Green
    "#E933FF",  # Hot Pink
    "#FFD333",  # Golden Yellow
    "#33FFD7",  # Aquamarine
    "#FF3366",  # Bright Red-Pink
    "#8A33FF",  # Purple
    "#33FF8A",  # Bright Mint
    "#FFB833",  # Orange-Yellow
    "#338AFF",  # Bright Blue
    "#FF33A1",  # Pink-Magenta
    "#A1FF33",  # Yellow-Green
    "#33FFD3",  # Turquoise
    "#FF3380",  # Coral Pink
    "#7FFF33",  # Chartreuse Green
    "#33D7FF",  # Light Blue
    "#D733FF",  # Purple Magenta
    "#FF9A33",  # Vibrant Orange
    "#33FF57",  # Jade Green
]


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
        self.unique_label: list[str] = []
        self.distance_matrix: np.ndarray = None
        self.distance_scale_coefficients: Tuple[float, float] = tuple([1, 1])

    def add_location(self, location: Coordinates):
        if not ENABLE_COORDINATES:
            return

        if location not in self.coordinate_list:
            self.coordinate_list.append(location)

    def road_exists(self, new_road: Road):
        if new_road in self.road_list:
            return True
        return False

    def add_road(
        self,
        start_name: str,
        end_name: str,
        style: str = None,
        width: str = None,
        color: str = None,
    ):
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
                self.add_road(start, end, style, width, color)

    def add_road(
        self,
        start: Coordinates,
        end: Coordinates,
        style: str = None,
        width: str = None,
        color: str = None,
    ):
        new_road = Road(start, end, style, width, color)
        if self.road_exists(new_road):
            return

        self.road_list.append(new_road)
        self.add_location(start)
        self.add_location(end)

    def clean_roads(self):
        self.road_list = []

    def calculate_distance_matrix(self):
        # Extract latitudes and longitudes as NumPy arrays
        latitudes = np.array([coord.latitude for coord in self.coordinate_list])
        longitudes = np.array([coord.longitude for coord in self.coordinate_list])

        # Stack latitudes and longitudes into a 2D array of shape (n, 2)
        points = np.vstack((latitudes, longitudes)).T

        # Calculate the distance matrix using broadcasting and vectorized operations
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        self.distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    def get_distance_matrix(self, normalize: bool = False):
        distance_matrix = self.distance_matrix.copy()

        if normalize:
            min_val = np.min(distance_matrix)
            max_val = np.max(distance_matrix)
            # Avoid division by zero if all values are the same
            if max_val > min_val:
                distance_matrix = (distance_matrix - min_val) / (max_val - min_val)
                self.distance_scale_coefficients = tuple([min_val, max_val])
            else:
                distance_matrix = np.zeros_like(distance_matrix)

        return distance_matrix

    def rescale_distance(self, normalized_distance: float):
        min_val, max_val = self.distance_scale_coefficients
        return normalized_distance * (max_val - min_val) + min_val

    def rescale_number_of_trucks(self, normalized_number_of_trucks: float):
        return int(normalized_number_of_trucks * (len(self.coordinate_list) - 1))

    def compose_visualization_coordinates(self):
        """
        Visualize the graph with coordinates as points.
        """

        customer_count = 1
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

            self.unique_label.append(label)

            if ENABLE_POINT_LABEL:
                label = ""
                if coordinate.name == DEPOT_NAME:
                    label = DEPOT_NAME
                else:
                    label = customer_count
                    customer_count += 1

                plt.text(
                    coordinate.latitude + 1,
                    coordinate.longitude + 1,
                    label,
                    fontsize=9,
                )

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

    def visualize(self, graph_title: str = None):
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")

        if graph_title:
            plt.title(f"Map Graph Visualization - {graph_title} - Map: {SEED}")
        else:
            plt.title(f"Map Graph Visualization - Map: {SEED}")

        plt.legend(loc="upper left")
        plt.xlim(RANGE_OF_MAP[0])
        plt.ylim(RANGE_OF_MAP[1])

        if ENABLE_GRID:
            plt.grid(True)

        # plt.tight_layout()
        plt.show()


class CustomRandomSampling(Sampling):
    """
    A sample has the following format:
    [4, 0, 1, 0, 6, 0, 3, 0, 5, 0, 2, 1]

    For example.

    The number of customers is 6.
    With a permutation of customer list is [4, 1, 6, 3, 5, 2]. Add a binary flag after each customer to indicate routes. The last flag always is 1.

    [4, 0, 1, 0, 6, 1, 3, 0, 5, 0, 2, 1] indicates that there are 2 routes
    [4, 1, 6] and [3, 5, 2].

    encoded routes: [4, 0, 1, 0, 6, 1, 3, 0, 5, 0, 2, 1]
    decoed routes: [[4, 1, 6], [3, 5, 2]]
    """

    def __init__(self, number_of_customer):
        super().__init__()
        self.number_of_truck = number_of_customer
        self.number_of_customer = number_of_customer

    def _insert_pairs_randomly(self, input_array, new_array):
        # Convert input array to pairs
        input_pairs = [
            (input_array[i], input_array[i + 1]) for i in range(0, len(input_array), 2)
        ]
        new_pairs = [
            (new_array[i], new_array[i + 1]) for i in range(0, len(new_array), 2)
        ]

        # Randomly shuffle the new pairs
        random.shuffle(new_pairs)

        # Insert each new pair into a random position in the original list
        for pair in new_pairs:
            insert_position = random.randint(
                0, len(input_pairs)
            )  # Allow insertion at the start, middle, or end
            input_pairs.insert(insert_position, pair)

        # Flatten the list back to a single array
        result = [item for pair in input_pairs for item in pair]
        return result

    def _add_zero_after_each_element(self, input_list):
        result = []
        for item in input_list:
            result.extend([item, 0])
        return result

    def generate_permutations_with_last_flag_one(self, input_list):
        # Extract all odd indices to determine the flag positions
        flag_indices = [i for i in range(len(input_list)) if i % 2 == 1]

        # Ensure the last flag is always 1
        flag_indices_except_last = flag_indices[:-1]
        last_flag_index = flag_indices[-1]

        # Generate all possible permutations for the other flag values (0 or 1)
        all_permutations = product([0, 1], repeat=len(flag_indices_except_last))

        # Generate lists based on the permutations
        result = []
        for perm in all_permutations:
            new_list = input_list[:]  # Make a copy of the original list
            for index, flag_value in zip(flag_indices_except_last, perm):
                new_list[index] = flag_value
            # Set the last flag to 1
            new_list[last_flag_index] = 1
            result.append(new_list)

        # Ensure the given list is included
        if input_list not in result:
            result.append(input_list)

        return result

    def _do(self, problem, n_samples, **kwargs):
        # Case 1: If this is HDP problem and it has NDP solutions, then use those solutions.
        if (
            isinstance(problem, HDP_MultiObjectiveVehicleRoutingProblem)
            and isinstance(problem.ndp_encoded_solutions, np.ndarray)
            and problem.ndp_encoded_solutions.shape[0] > 0
        ):
            initial_ndp_encoded_solutions = problem.ndp_encoded_solutions
            transformed_initial_ndp_encoded_solutions = []

            gap_in_number_of_customers = problem.number_of_hdp_customer - len(
                problem.ndp_customer_list
            )

            for index in range(len(initial_ndp_encoded_solutions)):
                # Create a new permutation of customers
                permutation = (
                    np.random.permutation(gap_in_number_of_customers)
                    + gap_in_number_of_customers
                    + 1
                )

                # Create a encoded array of new customers
                new_array = self._add_zero_after_each_element(permutation)

                # Create new sample from NDP solution and encoded array of new customers
                new_sample = self._insert_pairs_randomly(
                    initial_ndp_encoded_solutions[index], new_array
                )

                transformed_initial_ndp_encoded_solutions.extend(
                    self.generate_permutations_with_last_flag_one(new_sample)
                )

            return transformed_initial_ndp_encoded_solutions

        # Case 2: The following code is for NDP problem and HDP problem that does not has NDP solutions
        else:
            X = []  # Start with an empty list to hold samples
            part_size = (
                n_samples // self.number_of_customer
            )  # Calculate size of each part

            for i in range(self.number_of_customer):
                # Determine how many `1`s to place in sample[1::2] for this part
                num_ones = i

                for _ in range(part_size):
                    # Create a random permutation of customer indices
                    permutation = np.random.permutation(self.number_of_customer) + 1

                    # Initialize the sample array
                    sample = np.ones(2 * self.number_of_customer, dtype=int)

                    # Set customer indices at even positions
                    sample[0::2] = permutation

                    # Generate a list with `num_ones` 1s and the rest 0s
                    binary_choices = [1] * num_ones + [0] * (
                        sample[1::2].shape[0] - num_ones
                    )

                    # Randomly shuffle the binary choices
                    np.random.shuffle(binary_choices)

                    # Set the binary choices in sample[1::2]
                    sample[1::2] = binary_choices

                    # Ensure the last element is 1
                    sample[-1] = 1

                    # Add the generated sample to the list
                    X.append(sample)

            # Handle any leftover samples due to rounding
            remaining_samples = n_samples % self.number_of_customer
            for _ in range(remaining_samples):
                # Create a random permutation of customer indices
                permutation = np.random.permutation(self.number_of_customer) + 1

                # Initialize the sample array
                sample = np.ones(2 * self.number_of_customer, dtype=int)

                # Set customer indices at even positions
                sample[0::2] = permutation

                # Set all 0s in sample[1::2] for leftover samples
                sample[1::2] = [0] * sample[1::2].shape[0]

                # Ensure the last element is 1
                sample[-1] = 1

                # Add the generated sample to the list
                X.append(sample)

            return X


class Helper:
    """
    All form of solution (routes)

    Encoded form (List[int]): [3, 0, 1, 1, 5, 0, 4, 1, 2, 1]

    Flatted form (List[set[int, int]]): [{0, 3}, {3, 1}, {1, 0}, {0, 5}, {5, 4}, {4, 0}, {0, 2}, {2, 0}]

    Decoded form (List[List[int]]): [[3, 1], [5, 4], [2]]

    """

    def __init__(self):
        pass

    @staticmethod
    def transform_encoded_to_decoded(encoded_routes: List[int]):
        """
        Transform encoded routes to decoded routes.

        Args:
            encoded_routes (List[int]): [3, 0, 1, 1, 5, 0, 4, 1, 2, 1]

        Returns:
            decoded_routes (List[List[int]]): [[3, 1], [5, 4], [2]]
        """

        encoded_routes = copy.deepcopy(encoded_routes)
        encoded_routes[0::2] = np.argsort(np.argsort(encoded_routes[0::2])) + 1
        try:
            encoded_routes = np.array(encoded_routes).astype(int).tolist()
        except:
            pass

        decoded_routes: List = []
        current_group: List = []

        for i in range(0, len(encoded_routes), 2):
            # Append the even-index element to the current group
            current_group.append(encoded_routes[i])

            # Check if the next odd-index element is 1
            if i + 1 < len(encoded_routes) and encoded_routes[i + 1] == 1:
                # If it's a separator (1), add the current group to the result and start a new group
                decoded_routes.append(current_group)
                current_group = []

        # Append the last group if it's non-empty
        if current_group:
            decoded_routes.append(current_group)

        return decoded_routes

    @staticmethod
    def transform_decoded_to_encoded(decoded_routes: List[List[int]]) -> List[int]:
        """
        Transform decoded routes back to encoded routes.

        Args:
            decoded_routes (List[List[int]]): [[3, 1], [5, 4], [2]]

        Returns:
            encoded_routes (List[int]): [3, 0, 1, 1, 5, 0, 4, 1, 2, 1]
        """
        encoded_routes: List[int] = []

        for group in decoded_routes:
            for index, element in enumerate(group):
                # Add the element itself
                encoded_routes.append(element)
                # Add 1 after the last element in the group, 0 otherwise
                if index == len(group) - 1:
                    encoded_routes.append(1)
                else:
                    encoded_routes.append(0)

        # Convert the first half of even-indexed elements to their "ranked" form
        even_indices = np.array(encoded_routes[0::2])
        ranked_indices = np.argsort(np.argsort(even_indices)) + 1
        encoded_routes[0::2] = ranked_indices.tolist()

        return encoded_routes

    @staticmethod
    def transform_encoded_to_flatted(encoded_route: List[int]) -> List[set[int, int]]:
        """
        Transform encoded routes to flatted routes

        Args:
            encoded_routes (List[int]): [3, 0, 1, 1, 5, 0, 4, 1, 2, 1]

        Returns:
            flatted_routes (List[set[int, int]]): [{0, 3}, {3, 1}, {1, 0}, {0, 5}, {5, 4}, {4, 0}, {0, 2}, {2, 0}]
        """

        flatted_routes = []

        for i in range(0, len(encoded_route), 2):
            customer = encoded_route[i]
            next_customer = None

            flag = encoded_route[i + 1]

            if (i + 2) < len(encoded_route):
                next_customer = encoded_route[i + 2]

            if i == 0:
                flatted_routes.append({0, customer})

            if flag == 0 and next_customer:
                flatted_routes.append({customer, next_customer})

            elif flag == 1:
                flatted_routes.append({customer, 0})

                if next_customer:
                    flatted_routes.append({0, next_customer})

        return flatted_routes

    @staticmethod
    def transform_encoded_to_adjacent(
        encoded_route: List[int],
    ) -> Tuple[List[List[int]]]:
        """
        Transform encoded routes to adjacent routes

        Args:
            encoded_routes (List[int]): [3, 0, 1, 1, 5, 0, 4, 1, 2, 1]

        Returns:
            adjacent_routes (Tuple[List[List[int]]]): ([[3, 1], [1, 5], [5, 4], [4, 2], [2, 3]], [0, 1, 0, 1, 1])
        """
        encoded_route = copy.deepcopy(encoded_route)
        customers = np.argsort(encoded_route[0::2]) + 1
        flags = encoded_route[1::2]
        return (
            [
                [
                    customers[i],
                    (customers[i + 1] if i + 1 < len(customers) else customers[0]),
                ]
                for i in range(len(customers))
            ],
            flags,
        )

    @staticmethod
    def transform_adjacent_to_encoded(
        adjacent_routes: List[List[int]], flags: List[int] = None
    ) -> List[int]:
        """
        Transform adjacent routes back to encoded routes.

        Args:
            adjacent_routes (List[List[int]]): [[3, 1], [1, 5], [5, 4], [4, 2], [2, 3]]
            flags (List[int]) (optional): [0, 1, 0, 1, 1]

        Returns:
            encoded_routes (List[int]): [3, 0, 1, 1, 5, 0, 4, 1, 2, 1]
        """
        encoded_route = [0] * len(adjacent_routes) * 2
        customers = []
        for i, (start, end) in enumerate(adjacent_routes):
            customers.append(start)

        encoded_route[0::2] = customers
        if isinstance(flags, np.ndarray) and flags.size > 0:
            if len(flags) != len(customers):
                raise ValueError(
                    f"Invalid number of flags: must be {len(customers)}, got {len(flags)}."
                )

            encoded_route[1::2] = flags

        encoded_route = [int(x) for x in encoded_route]

        return encoded_route

    @staticmethod
    def calculate_x_lower_bound(number_of_customer: int) -> np.array:
        lower_bound = np.zeros(number_of_customer * 2, dtype=int)

        # Set 1 at all even indices (customer)
        lower_bound[0::2] = 1

        # Set 0 at all odd indices (flag indicate boundary of a route)
        lower_bound[1::2] = 0

        return lower_bound

    @staticmethod
    def calculate_x_upper_bound(number_of_customer) -> np.array:
        upper_bound = np.zeros(number_of_customer * 2, dtype=int)

        # Set number of NDP customers at all even indices (customer)
        upper_bound[0::2] = number_of_customer

        # Set 1 at all odd indices (flag indicate boundary of a route)
        upper_bound[1::2] = 1

        # Last flag always is 1
        upper_bound[-1] = 1

        return upper_bound

    @staticmethod
    def calculate_max_distance_among_trucks(
        matrix_distance: np.ndarray,
        decoded_routes: List[List[int]],
    ) -> float:
        # Convert values of decoded_routes to int
        decoded_routes = [
            [int(value) for value in sublist] for sublist in decoded_routes
        ]

        # Initialize max distance
        max_distance_among_trucks = 0

        # Calculate the maximum distance for each truck's route and update max_distance_among_trucks
        for route in decoded_routes:
            # Calculate the distance from the depot to the first customer
            route_distance = matrix_distance[0, route[0]]

            # Calculate total distance for this truck route
            route_distance += sum(
                matrix_distance[route[i], route[i + 1]] for i in range(len(route) - 1)
            )

            # Calculate the distance from the last customer to the depot
            route_distance += matrix_distance[route[-1], 0]

            # Update max_distance_among_trucks if this route's distance is larger
            max_distance_among_trucks = max(max_distance_among_trucks, route_distance)

        return max_distance_among_trucks


    @staticmethod
    def calculate_similarity(
        matrix_distance: np.ndarray,
        decoded_routes: List[List[int]],
    ) -> float:
        # Convert values of decoded_routes to int
        decoded_routes = [
            [int(value) for value in sublist] for sublist in decoded_routes
        ]

        # Initialize max distance
        max_distance_among_trucks = 0

        # Calculate the maximum distance for each truck's route and update max_distance_among_trucks
        for route in decoded_routes:
            # Calculate the distance from the depot to the first customer
            route_distance = matrix_distance[0, route[0]]

            # Calculate total distance for this truck route
            route_distance += sum(
                matrix_distance[route[i], route[i + 1]] for i in range(len(route) - 1)
            )

            # Calculate the distance from the last customer to the depot
            route_distance += matrix_distance[route[-1], 0]

            # Update max_distance_among_trucks if this route's distance is larger
            max_distance_among_trucks = max(max_distance_among_trucks, route_distance)

        return max_distance_among_trucks

    @staticmethod
    def calculate_similarity_between_flatted_routes(
        flatted_route_1: List[set[int, int]],
        flatted_route_2: List[set[int, int]],
        weight_coefficient: float = 1.0,
    ):

        if weight_coefficient == 0:
            raise ValueError("Cannot divide by zero")

        similarity = (
            len(
                set(map(frozenset, flatted_route_1))
                & set(map(frozenset, flatted_route_2))
            )
            / weight_coefficient
        )

        return similarity

    @staticmethod
    def calculate_similarity_between_hdp_flatted_routes_and_ndp_flatted_routes_list(
        hdp_flatted_routes: List[set[int, int]],
        ndp_flatted_routes_list: List[List[set[int, int]]],
        return_full_result: bool = True,
    ):
        max_similarity = 0.0
        best_ndp_flatted_routes = ndp_flatted_routes_list[0]

        for ndp_flatted_routes in ndp_flatted_routes_list:
            similarity = Helper.calculate_similarity_between_flatted_routes(
                hdp_flatted_routes, ndp_flatted_routes, len(ndp_flatted_routes)
            )
            if similarity > max_similarity:
                max_similarity = similarity
                best_ndp_flatted_routes = ndp_flatted_routes

        if return_full_result:
            return max_similarity

        return set(hdp_flatted_routes, best_ndp_flatted_routes, max_similarity)

    @staticmethod
    def calculate_similarity_between_encoded_routes(
        encoded_route_1: List[int],
        encoded_route_2: List[int],
        weight_coefficient: float = 1.0,
    ):

        if weight_coefficient == 0:
            raise ValueError("Cannot divide by zero")

        flatted_route_1 = Helper.transform_encoded_to_flatted(encoded_route_1)
        flatted_route_2 = Helper.transform_encoded_to_flatted(encoded_route_2)

        return Helper.calculate_similarity_between_flatted_routes(
            flatted_route_1, flatted_route_2, len(flatted_route_2)
        )

    @staticmethod
    def calculate_similarity_between_hdp_encoded_routes_and_ndp_encoded_routes_list(
        hdp_encoded_routes: List[List[int]],
        ndp_encoded_routes_list: List[List[int]],
        return_full_result: bool = True,
    ):
        hdp_flatted_routes = Helper.transform_encoded_to_flatted(hdp_encoded_routes)

        max_similarity = 0.0
        best_ndp_flatted_routes = Helper.transform_encoded_to_flatted(
            ndp_encoded_routes_list[0]
        )

        for ndp_encoded_routes in ndp_encoded_routes_list:
            ndp_flatted_routes = Helper.transform_encoded_to_flatted(ndp_encoded_routes)
            similarity = Helper.calculate_similarity_between_flatted_routes(
                hdp_flatted_routes, ndp_flatted_routes, len(ndp_flatted_routes)
            )
            if similarity > max_similarity:
                max_similarity = similarity
                best_ndp_flatted_routes = ndp_flatted_routes

        if not return_full_result:
            return max_similarity

        return tuple(
            [
                Helper.transform_encoded_to_decoded(hdp_encoded_routes),
                Helper.transform_encoded_to_decoded(best_ndp_flatted_routes),
                max_similarity,
            ]
        )


class EdgeExchangeCrossover(Crossover, Helper):

    def __init__(self, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)

    @staticmethod
    def is_valid_adjacent_route(adjacent_route: List[List[int]]):
        if not adjacent_route:
            return False  # Empty list is not valid

        # Check if the tail of the previous element matches the head of the next element
        for i in range(len(adjacent_route) - 1):
            if adjacent_route[i][1] != adjacent_route[i + 1][0]:
                return False

        # Check if the head of the first element matches the tail of the last element
        if adjacent_route[0][0] != adjacent_route[-1][1]:
            return False

        counts = Counter(num for sublist in adjacent_route for num in sublist)

        # Ensure each number appears at most twice
        if any(count > 2 for count in counts.values()):
            return False

        return True

    @staticmethod
    def reverse_and_swap(lst, a, b):
        """
        Reverse and swap all elements in a list between indices a and b (exclusive),
        treating the list as cyclic and keeping the list length unchanged.

        Args:
            lst (list): A list of continuous pairs (e.g., [[1, 2], [2, 3], [3, 4]...]).
            a (int): Start index (exclusive).
            b (int): End index (exclusive).

        Returns:
            list: The modified list.
        """
        if not isinstance(lst, list) or len(lst) < 0:
            raise ValueError(f"lst ({lst}) must be a list.")

        for idx, var in zip(("a", "b"), (a, b)):
            if var < 0 or var >= len(lst):
                raise ValueError(f"{idx} ({var}) is invalid.")

        if a == b:
            raise ValueError(f"a ({a}) and b ({b}) must be different.")

        lst = copy.deepcopy(lst)

        # Interal reverse and swap
        if a < b:
            sublist = lst[a + 1 : b]
            reversed_sublist = [[item[1], item[0]] for item in sublist]
            reversed_sublist.reverse()
            lst[a + 1 : b] = reversed_sublist

        # External reverse and swap
        else:
            first_part_length = len(lst) - (a + 1)

            sublist = lst[a + 1 :]
            sublist.extend(lst[:b])
            reversed_sublist = [[item[1], item[0]] for item in sublist]
            reversed_sublist.reverse()

            lst[a + 1 :] = reversed_sublist[:first_part_length]
            lst[:b] = reversed_sublist[first_part_length:]

        return lst

    @staticmethod
    def swap_elements(list1, index1, list2, index2):
        """
        Swap elements between two lists at the specified indices.

        Parameters:
        - list1: First list
        - list2: Second list
        - index1: Index of the element in list1 to swap
        - index2: Index of the element in list2 to swap
        """
        # Swap elements
        list1[index1], list2[index2] = list2[index2], list1[index1]

    @staticmethod
    def edge_exchange_crossover(
        initial_adjacent_route_1: List[List[int]],
        initial_adjacent_route_2: List[List[int]],
    ):
        if not EdgeExchangeCrossover.is_valid_adjacent_route(initial_adjacent_route_1):
            raise ValueError(
                f"Invalid initial_adjacent_route_1 {initial_adjacent_route_1}."
            )

        if not EdgeExchangeCrossover.is_valid_adjacent_route(initial_adjacent_route_2):
            raise ValueError(
                f"Invalid initial_adjacent_route_2 {initial_adjacent_route_2}."
            )

        adjacent_route_1 = copy.deepcopy(initial_adjacent_route_1)
        adjacent_route_2 = copy.deepcopy(initial_adjacent_route_2)

        i1_index = random.randint(0, len(adjacent_route_1) - 1)
        i1 = adjacent_route_1[i1_index]

        # Finding i2 (same head as i1)
        i2_index = next(
            (adjacent_route_2.index(x) for x in adjacent_route_2 if x[0] == i1[0]), None
        )
        i2 = adjacent_route_2[i2_index]

        exist_list = [False] * len(initial_adjacent_route_1)
        while np.array_equal(i1, i2):
            i1_index = random.randint(0, len(adjacent_route_1) - 1)
            exist_list[i1_index] = True

            i1 = adjacent_route_1[i1_index]

            # Finding i2 (same head as i1)
            i2_index = next(
                (adjacent_route_2.index(x) for x in adjacent_route_2 if x[0] == i1[0]),
                None,
            )
            i2 = adjacent_route_2[i2_index]

            # In case all pairs of initial_adjacent_route_1 and initial_adjacent_route_2 are the same, return both
            if all(exist_list):
                return initial_adjacent_route_1, initial_adjacent_route_2

        # Finding j2 (same tail as i1)
        j2_index = next(
            (adjacent_route_2.index(x) for x in adjacent_route_2 if x[0] == i1[1]),
            None,
        )

        # Finding j1 (same head as j1)
        j1_index = next(
            (adjacent_route_1.index(x) for x in adjacent_route_1 if x[0] == i2[1]), None
        )

        while True:
            i1 = adjacent_route_1[i1_index]
            i2 = adjacent_route_2[i2_index]

            # Finding j2 (same tail as i1)
            j2_index = next(
                (adjacent_route_2.index(x) for x in adjacent_route_2 if x[0] == i1[1]),
                None,
            )

            # Finding j1 (same head as j1)
            j1_index = next(
                (adjacent_route_1.index(x) for x in adjacent_route_1 if x[0] == i2[1]),
                None,
            )

            # Swap i1 and i2
            EdgeExchangeCrossover.swap_elements(
                adjacent_route_1, i1_index, adjacent_route_2, i2_index
            )

            adjacent_route_1 = EdgeExchangeCrossover.reverse_and_swap(
                adjacent_route_1, i1_index, j1_index
            )
            adjacent_route_2 = EdgeExchangeCrossover.reverse_and_swap(
                adjacent_route_2, i2_index, j2_index
            )

            if EdgeExchangeCrossover.is_valid_adjacent_route(
                adjacent_route_1
            ) and EdgeExchangeCrossover.is_valid_adjacent_route(adjacent_route_2):
                break

            i1_index = j1_index
            i2_index = j2_index

        return adjacent_route_1, adjacent_route_2

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape

        Q = [[], []]

        for index in range(n_matings):
            parent_1 = X[0][index]
            parent_2 = X[1][index]

            # Avoid duplicates
            while np.array_equal(parent_1, parent_2):
                index = random.randint(0, n_matings - 1)
                parent_2 = X[1][index]

            # Crossover
            adjacent_form_parent_1, flag_parent_1 = self.transform_encoded_to_adjacent(
                parent_1
            )
            adjacent_form_parent_2, flag_parent_2 = self.transform_encoded_to_adjacent(
                parent_2
            )

            offspring_1, offspring_2 = self.edge_exchange_crossover(
                adjacent_form_parent_1, adjacent_form_parent_2
            )

            Q[0].append(self.transform_adjacent_to_encoded(offspring_1, flag_parent_1))
            Q[1].append(self.transform_adjacent_to_encoded(offspring_2, flag_parent_2))

        Q = np.array(Q)

        return Q


class EXX(EdgeExchangeCrossover):
    pass


class OrderSplitMutation(Mutation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = 1

    @staticmethod
    def order_mutation(x):
        total_truck = sum(x[1::2])

        if total_truck == 0:
            raise ValueError(f"Total truck ({total_truck}) must greater than zero.")

        chosed_truck = random.randint(1, total_truck) - 1

        sublist_start_indices = [0]

        for index in range(len(x[1::2])):
            if x[1::2][index] == 1:
                sublist_start_indices.append((index + 1) * 2)

        sublist = x[
            sublist_start_indices[chosed_truck] : sublist_start_indices[
                chosed_truck + 1
            ]
        ]
        sublist_to_reverse: List = sublist[0::2]
        sublist_to_reverse = sublist_to_reverse[::-1]
        sublist[0::2] = sublist_to_reverse

        x[
            sublist_start_indices[chosed_truck] : sublist_start_indices[
                chosed_truck + 1
            ]
        ] = sublist

        return x

    @staticmethod
    def split_mutation(x):
        # Last flag always is 1
        number_of_truck_can_be_shuffle = sum(x[1::2]) - 1

        x[1::2][:-1] = 0

        ones_indices = random.sample(
            range(len(x[1::2][:-1])), number_of_truck_can_be_shuffle
        )

        for index in ones_indices:
            x[1::2][:-1][index] = 1

        return x

    def _do(self, problem, X, params=None, **kwargs):
        Xp = []
        for x in X:
            x[-1] = 1
            x = self.order_mutation(x)
            x = self.split_mutation(x)
            Xp.append(x)

        Xp = np.array(Xp)

        return Xp


class OSM(OrderSplitMutation):
    pass


class NDP_MultiObjectiveVehicleRoutingProblem(ElementwiseProblem, Helper):
    def __init__(
        self,
        number_of_ndp_customer: int,
        range_of_ndp_customer: Tuple[Tuple[int, int], Tuple[int, int]],
    ):

        self.number_of_ndp_customer = number_of_ndp_customer
        self.range_of_ndp_customer = range_of_ndp_customer

        self.depot: Depot = Depot(DEPOT_LOCATION[0], DEPOT_LOCATION[1])
        self.ndp_customer_list: list[NDP_Customer] = []

        print("\n\nNDP_MultiObjectiveVehicleRoutingProblem")

        # Define map
        self.define_map()

        # Calculate distance between coordinates (matrix)
        self.map_graph.calculate_distance_matrix()

        # Normalize objective
        self.normalized_distance_matrix = self.map_graph.get_distance_matrix(
            normalize=True
        )

        xl: np.array = self.calculate_x_lower_bound(self.number_of_ndp_customer)
        xu: np.array = self.calculate_x_upper_bound(self.number_of_ndp_customer)

        # Define problem
        super().__init__(
            n_var=self.number_of_ndp_customer * 2, n_obj=2, n_constr=0, xl=xl, xu=xu
        )

    def _evaluate(self, x: np.array, out, *args, **kwargs):
        number_of_truck = np.round(x[1::2]).astype(int).sum()
        encoded_routes = x

        decoded_routes = self.transform_encoded_to_decoded(encoded_routes)

        # Objective 1: Maximum distance traveled by any truck (normalized)
        f1 = self.calculate_max_distance_among_trucks(
            self.normalized_distance_matrix, decoded_routes
        )

        # Objective 2: Minimize the number of trucks used (normalized)
        f2 = number_of_truck / self.number_of_ndp_customer

        out["F"] = np.array([f1, f2])

    def define_map(self):
        # Add NDP locations
        for i in range(self.number_of_ndp_customer):
            self.ndp_customer_list.append(
                NDP_Customer(
                    latitude=round(
                        random.uniform(
                            self.range_of_ndp_customer[0][0],
                            self.range_of_ndp_customer[0][1],
                        ),
                        4,
                    ),
                    longitude=round(
                        random.uniform(
                            self.range_of_ndp_customer[1][0],
                            self.range_of_ndp_customer[1][1],
                        ),
                        4,
                    ),
                    name=f"{NDP_CUSTOMER_NAME}_{i+1}",
                )
            )

        self.map_graph = MapGraph()

        # Add depot
        self.map_graph.add_location(self.depot)

        # Add NDP customers
        for customer in self.ndp_customer_list:
            self.map_graph.add_location(customer)

    def get_map_graph(self):
        return self.map_graph

    def get_ndp_customer_list(self):
        return self.ndp_customer_list

    def visualize(self):
        plt.figure(figsize=FIG_SIZE)

        self.map_graph.compose_visualization_coordinates()
        self.map_graph.compose_visualization_roads()
        self.map_graph.visualize()


class HDP_MultiObjectiveVehicleRoutingProblem(ElementwiseProblem, Helper):
    def __init__(
        self,
        ndp_customer_list: List[Customer],
        number_of_hdp_customer: int,
        range_of_hdp_customer: Tuple[Tuple[int, int], Tuple[int, int]],
        ndp_encoded_solutions: np.array = None,
        optimize_similality: bool = False,
    ):
        self.ndp_customer_list = copy.deepcopy(ndp_customer_list)
        self.hdp_customer_list = copy.deepcopy(ndp_customer_list)
        self.number_of_hdp_customer = number_of_hdp_customer
        self.range_of_hdp_customer = range_of_hdp_customer
        self.optimize_similality = optimize_similality

        self.depot: Depot = Depot(DEPOT_LOCATION[0], DEPOT_LOCATION[1])

        if (
            isinstance(ndp_encoded_solutions, np.ndarray)
            and ndp_encoded_solutions.shape[0] > 0
        ):
            self.ndp_encoded_solutions = copy.deepcopy(ndp_encoded_solutions)

            if optimize_similality:
                print(
                    "\n\nDependent HDP_MultiObjectiveVehicleRoutingProblem - 3 objectives"
                )
            else:
                print(
                    "\n\nDependent HDP_MultiObjectiveVehicleRoutingProblem - 2 objectives"
                )
        else:
            self.ndp_encoded_solutions = None
            print("\n\nIndependent HDP_MultiObjectiveVehicleRoutingProblem")

        # Define map
        self.define_map()

        # Calculate distance between coordinates (matrix)
        self.map_graph.calculate_distance_matrix()

        # Normalize objective
        self.normalized_distance_matrix = self.map_graph.get_distance_matrix(
            normalize=True
        )

        xl: np.array = self.calculate_x_lower_bound(self.number_of_hdp_customer)
        xu: np.array = self.calculate_x_upper_bound(self.number_of_hdp_customer)

        n_obj = 2
        if self.optimize_similality:
            self.sample_count = 0
            n_obj = 3

        # Define problem
        super().__init__(
            n_var=self.number_of_hdp_customer * 2, n_obj=n_obj, n_constr=0, xl=xl, xu=xu
        )

    def _evaluate(self, x: np.array, out, *args, **kwargs):
        number_of_truck = np.round(x[1::2]).astype(int).sum()
        encoded_routes = x

        decoded_routes = self.transform_encoded_to_decoded(encoded_routes)

        # Objective 1: Maximum distance traveled by any truck (normalized)
        f1 = self.calculate_max_distance_among_trucks(
            self.normalized_distance_matrix, decoded_routes
        )

        # Objective 2: Minimize the number of trucks used (normalized)
        f2 = number_of_truck / self.number_of_hdp_customer

        if (
            self.ndp_encoded_solutions is not None
            and self.ndp_encoded_solutions.shape[0] > 0
            and self.optimize_similality
        ):
            f3 = (
                -1
                * self.calculate_similarity_between_hdp_encoded_routes_and_ndp_encoded_routes_list(
                    x, self.ndp_encoded_solutions, return_full_result=False
                )
            )
            self.sample_count += 1

            out["F"] = np.array([f1, f2, f3])

        else:
            out["F"] = np.array([f1, f2])

    def define_map(self):
        if len(self.hdp_customer_list) < self.number_of_hdp_customer:
            # Add HDP customers
            for i in range(self.number_of_hdp_customer - len(self.ndp_customer_list)):
                self.hdp_customer_list.append(
                    HDP_Customer(
                        latitude=round(
                            random.uniform(
                                self.range_of_hdp_customer[0][0],
                                self.range_of_hdp_customer[0][1],
                            ),
                            4,
                        ),
                        longitude=round(
                            random.uniform(
                                self.range_of_hdp_customer[1][0],
                                self.range_of_hdp_customer[1][1],
                            ),
                            4,
                        ),
                        name=f"{HDP_CUSTOMER_NAME}_{i+1}",
                    )
                )

        self.map_graph = MapGraph()

        # Add depot
        self.map_graph.add_location(self.depot)

        # Add HDP customers
        for customer in self.hdp_customer_list:
            self.map_graph.add_location(customer)

    def get_map_graph(self):
        return self.map_graph

    def get_hdp_customer_list(self):
        return self.hdp_customer_list

    def visualize(self):
        plt.figure(figsize=FIG_SIZE)

        self.map_graph.compose_visualization_coordinates()
        self.map_graph.compose_visualization_roads()
        self.map_graph.visualize()


class SolutionHandler(Helper):
    def __init__(self, map_graph: MapGraph):
        self.map_graph: MapGraph = map_graph
        self.result: Result = None

    def set_result(self, result: Result):
        self.result = copy.deepcopy(result)

    def _validate_number_of_solution_value(self, index_of_solution):
        if not self.result:
            raise ValueError("No solution found.")

        try:
            if (
                isinstance(index_of_solution, float)
                and index_of_solution > 0
                and index_of_solution < 1
            ):
                index_of_solution = int(len(self.result.X) * index_of_solution)

            _ = self.result.X[index_of_solution]

            return index_of_solution
        except:
            raise ValueError(f"'index_of_solution' is not valid.")

    def get_best_encoded_solutions(self, number_of_solutions: int | float = None):
        if not number_of_solutions:
            best_encoded_solutions = copy.deepcopy(self.result.X)
        else:
            number_of_solutions = self._validate_number_of_solution_value(
                number_of_solutions
            )
            best_encoded_solutions = copy.deepcopy(self.result.X[:number_of_solutions])

        result = np.zeros_like(best_encoded_solutions, dtype=int)

        for index in range(len(best_encoded_solutions)):
            result[index][0::2] = (
                np.argsort(np.argsort(best_encoded_solutions[index][0::2])) + 1
            )

            result[index][1::2] = np.round(best_encoded_solutions[index][1::2])

        return result

    def get_best_decoded_solutions(self, number_of_solutions: int | float = None):
        solution_list = [
            self.transform_encoded_to_decoded(solution)
            for solution in self.get_best_encoded_solutions(number_of_solutions)
        ]
        unique_nested_list = list(
            map(list, {tuple(map(tuple, sublist)) for sublist in solution_list})
        )

        # Convert the tuples back to lists
        return [list(map(list, item)) for item in unique_nested_list]

    def get_best_f(self, number_of_f: int = None):
        if not number_of_f:
            return copy.deepcopy(self.result.F)
        else:
            number_of_f = self._validate_number_of_solution_value(number_of_f)
            return copy.deepcopy(self.result.F[:number_of_f])

    def print_best_decoded_solutions(self, number_of_solutions: int = None):
        encoded_solution_list = self.get_best_encoded_solutions(number_of_solutions)
        f_list = self.get_best_f(number_of_solutions)

        for index in range(len(encoded_solution_list)):
            solution = copy.deepcopy(encoded_solution_list[index])
            solution = self.transform_encoded_to_decoded(solution)

            print()
            print(f"SOLUTION {index}")
            print(f"{solution}")
            print(
                f"- Maximum length among trucks: {self.map_graph.rescale_distance(f_list[index][0])}"
            )
            print(
                f"- Number of trucks used: {self.map_graph.rescale_number_of_trucks(f_list[index][1])}"
            )

    def print_similarity(
        self,
        encoded_hdp_solution: List[int],
        encoded_ndp_solution_list: List[int],
    ):
        similarity = self.calculate_similarity_between_hdp_encoded_routes_and_ndp_encoded_routes_list(
            hdp_encoded_routes=encoded_hdp_solution,
            ndp_encoded_routes_list=encoded_ndp_solution_list,
            return_full_result=False,
        )
        print(f"Similarity: {similarity}")

    def visualize_solution(self, graph_title: str = None, index_of_solution: int = 0):
        self._validate_number_of_solution_value(index_of_solution)

        encoded_solution: np.array = self.result.X[index_of_solution]

        route_list: List[List[int]] = self.transform_encoded_to_decoded(
            encoded_solution
        )

        # Add roads
        self.map_graph.clean_roads()
        coordinate_list = self.map_graph.coordinate_list
        depot = coordinate_list[0]

        for route_index in range(len(route_list)):
            color = ROUTE_COLORS[route_index]
            self.map_graph.add_road(
                start=depot,
                end=coordinate_list[route_list[route_index][0]],
                width=2,
                color=color,
            )

            for index in range(len(route_list[route_index]) - 1):
                self.map_graph.add_road(
                    coordinate_list[route_list[route_index][index]],
                    coordinate_list[route_list[route_index][index + 1]],
                    width=2,
                    color=color,
                )

            self.map_graph.add_road(
                coordinate_list[route_list[route_index][-1]],
                depot,
                width=2,
                color=color,
            )

        plt.figure(figsize=FIG_SIZE)
        self.map_graph.compose_visualization_coordinates()
        self.map_graph.compose_visualization_roads()
        self.map_graph.visualize(graph_title)


class LocalSearchHandler(SolutionHandler):
    def __init__(self, map_graph: MapGraph):
        print(
            "\n\nDependent HDP_MultiObjectiveVehicleRoutingProblem - 2 objectives - memetic"
        )
        self.map_graph: MapGraph = map_graph
        self.result: Result = None
        self.hdp_decoded_solution_list: List[List[int]] = None

    def set_result(self, result: Result):
        self.result = copy.deepcopy(result)

    def _find_edges(self, solution):
        edges = []
        for car in solution:
            if len(car) < 2:
                continue
            else:
                for i in range(len(car) - 1):
                    edge = [car[i], car[i + 1]]
                    edges.append(edge)
        return edges

    def _find_customer(self, solution, customer):
        for i in range(len(solution)):
            car = solution[i]
            if customer in car:
                return i, car.index(customer)
        raise ValueError("Customer not found.")

    def _find_customer_after_first_in_hdp_solution(
        self, hdp_solution, edge, force_same_car=True
    ):
        first_customer_car_index, first_customer_position_index = self._find_customer(
            solution=hdp_solution, customer=edge[0]
        )
        # Normal: in the same car
        if (
            len(hdp_solution[first_customer_car_index])
            > first_customer_position_index + 1
        ):
            return first_customer_car_index, first_customer_position_index + 1
        # If in the next car, get next position as usual
        elif not force_same_car:
            # get the first of next car
            if len(hdp_solution) > first_customer_car_index + 1:
                return first_customer_car_index + 1, 0
            # If in the last car already, get the first car
            else:
                return 0, 0
        else:
            return None, None

    def _local_search_single_edge(self, hdp_solution, edge, force_same_car=True):
        hdp_solution = copy.deepcopy(hdp_solution)
        first_customer = edge[0]
        second_customer = edge[1]
        customer_after_first_car_index, customer_after_first_customer_index = (
            self._find_customer_after_first_in_hdp_solution(
                hdp_solution=hdp_solution, edge=edge, force_same_car=force_same_car
            )
        )

        if (
            customer_after_first_car_index is not None
            and customer_after_first_customer_index is not None
        ):
            customer_after_first = hdp_solution[customer_after_first_car_index][
                customer_after_first_customer_index
            ]
            second_customer_car_index, second_customer_position_index = (
                self._find_customer(solution=hdp_solution, customer=second_customer)
            )
            hdp_solution[customer_after_first_car_index][
                customer_after_first_customer_index
            ] = second_customer
            hdp_solution[second_customer_car_index][
                second_customer_position_index
            ] = customer_after_first

        return hdp_solution

    def _internal_calculate_local_search(
        self, ndp_solution, hdp_solution, force_same_car=True
    ):
        edges = self._find_edges(ndp_solution)
        for edge in edges:
            hdp_solution = self._local_search_single_edge(
                hdp_solution=hdp_solution, edge=edge, force_same_car=force_same_car
            )
        return hdp_solution

    def get_best_encoded_solutions(self, number_of_solutions: int | float = None):
        solution_list = self.result.X

        if self.hdp_decoded_solution_list:
            solution_list = [
                self.transform_decoded_to_encoded(hdp_decoded_solution)
                for hdp_decoded_solution in self.hdp_decoded_solution_list
            ]

        if not number_of_solutions:
            best_encoded_solutions = copy.deepcopy(solution_list)
        else:
            number_of_solutions = self._validate_number_of_solution_value(
                number_of_solutions
            )
            best_encoded_solutions = copy.deepcopy(solution_list[:number_of_solutions])

        result = np.zeros_like(best_encoded_solutions, dtype=int)

        for index in range(len(best_encoded_solutions)):
            result[index][0::2] = (
                np.argsort(np.argsort(best_encoded_solutions[index][0::2])) + 1
            )

            result[index][1::2] = np.round(best_encoded_solutions[index][1::2])

        return result

    def calculate_local_search(
        self,
        ndp_encoded_solution_list: List[List[int]],
    ):
        if (
            not isinstance(ndp_encoded_solution_list, np.ndarray)
            or ndp_encoded_solution_list.shape[0] < 1
        ):
            raise ValueError(
                "'ndp_encoded_solution_list' must be a list with at least one element."
            )

        hdp_encoded_solution_list = self.get_best_encoded_solutions()
        ndp_encoded_solution_list = copy.deepcopy(ndp_encoded_solution_list)

        self.hdp_decoded_solution_list: List[List[int]] = []

        for hdp_encoded_solution in hdp_encoded_solution_list:

            # Gather set of hdp decoded solution and corresponding most similar ndp decoded solution
            similar_tuple = self.calculate_similarity_between_hdp_encoded_routes_and_ndp_encoded_routes_list(
                hdp_encoded_routes=hdp_encoded_solution,
                ndp_encoded_routes_list=ndp_encoded_solution_list,
                return_full_result=True,
            )

            # Apply local search
            local_search_tuple = self._internal_calculate_local_search(
                hdp_solution=similar_tuple[0],
                ndp_solution=similar_tuple[1],
                force_same_car=True,
            )

            self.hdp_decoded_solution_list.append(local_search_tuple)

    def get_hdp_decoded_solution_list(self, number_of_solutions: int = None):
        if not number_of_solutions:
            return self.hdp_decoded_solution_list
        else:
            self._validate_number_of_solution_value(number_of_solutions)
        return self.hdp_decoded_solution_list[:number_of_solutions]

    def print_best_decoded_solutions(self, number_of_solutions: int = None):
        decoded_solution_list = self.get_hdp_decoded_solution_list(number_of_solutions)
        f_list = self.get_best_f(number_of_solutions)

        for index in range(len(decoded_solution_list)):
            solution = copy.deepcopy(decoded_solution_list[index])

            print()
            print(f"SOLUTION {index}")
            print(f"{solution}")
            print(
                f"- Maximum length among trucks: {self.map_graph.rescale_distance(f_list[index][0])}"
            )
            print(
                f"- Number of trucks used: {self.map_graph.rescale_number_of_trucks(f_list[index][1])}"
            )

    def visualize_solution(self, graph_title: str = None, index_of_solution: int = 0):
        self._validate_number_of_solution_value(index_of_solution)

        route_list: List[List[int]] = self.hdp_decoded_solution_list[index_of_solution]

        # Add roads
        self.map_graph.clean_roads()
        coordinate_list = self.map_graph.coordinate_list
        depot = coordinate_list[0]

        for route_index in range(len(route_list)):
            color = ROUTE_COLORS[route_index]
            self.map_graph.add_road(
                start=depot,
                end=coordinate_list[route_list[route_index][0]],
                width=2,
                color=color,
            )

            for index in range(len(route_list[route_index]) - 1):
                self.map_graph.add_road(
                    coordinate_list[route_list[route_index][index]],
                    coordinate_list[route_list[route_index][index + 1]],
                    width=2,
                    color=color,
                )

            self.map_graph.add_road(
                coordinate_list[route_list[route_index][-1]],
                depot,
                width=2,
                color=color,
            )

        plt.figure(figsize=FIG_SIZE)
        self.map_graph.compose_visualization_coordinates()
        self.map_graph.compose_visualization_roads()
        self.map_graph.visualize(graph_title)


def main():
    """
    NDP problem
    """
    ndp_problem = NDP_MultiObjectiveVehicleRoutingProblem(
        number_of_ndp_customer=NUMBER_OF_NDP_CUSTOMER,
        range_of_ndp_customer=RANGE_OF_NDP_CUSTOMER,
    )

    # ndp_problem.visualize()

    ndp_algorithm = NSGA2(
        pop_size=30,
        n_offsprings=10,
        sampling=CustomRandomSampling(NUMBER_OF_NDP_CUSTOMER),
        # crossover=SBX(prob=0.9, eta=15),
        crossover=EXX(),
        # mutation=PM(eta=20),
        mutation=OSM(),
        eliminate_duplicates=True,
    )

    # Run the optimization
    ndp_res = minimize(ndp_problem, ndp_algorithm, ("n_gen", 200), verbose=False)

    # Create solution handler
    ndp_solution_handler = SolutionHandler(ndp_problem.get_map_graph())
    ndp_solution_handler.set_result(ndp_res)
    ndp_solution_handler.print_best_decoded_solutions(1)
    if ENABLE_SOLUTION_VISUALIZATION:
        ndp_solution_handler.visualize_solution("NDP problem")

    """
    HDP problem without solution from NDP (independent HDP problem)
    """
    if ENABLE_IND_HDP_PROBLEM:
        ind_hdp_problem = HDP_MultiObjectiveVehicleRoutingProblem(
            ndp_customer_list=ndp_problem.get_ndp_customer_list(),
            number_of_hdp_customer=NUMBER_OF_HDP_CUSTOMER,
            range_of_hdp_customer=RANGE_OF_HDP_CUSTOMER,
        )

        # ind_hdp_problem.visualize()

        ind_hdp_algorithm = NSGA2(
            pop_size=1000,
            n_offsprings=20,
            sampling=CustomRandomSampling(NUMBER_OF_HDP_CUSTOMER),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        # Run the optimization
        ind_hdp_res = minimize(
            ind_hdp_problem, ind_hdp_algorithm, ("n_gen", 200), verbose=False
        )

        # Create solution handler
        ind_hdp_solution_handler = SolutionHandler(ind_hdp_problem.get_map_graph())
        ind_hdp_solution_handler.set_result(ind_hdp_res)
        ind_hdp_solution_handler.print_best_decoded_solutions(1)

        if ENABLE_SOLUTION_VISUALIZATION:
            ind_hdp_solution_handler.visualize_solution("Independent HDP problem")

        ind_hdp_solution_handler.print_similarity(
            encoded_hdp_solution=ind_hdp_solution_handler.get_best_encoded_solutions(1)[
                0
            ],
            encoded_ndp_solution_list=ndp_solution_handler.get_best_encoded_solutions(),
        )

    """
    HDP problem with initial NDP solutions (2 objectives)
    """
    if ENABLE_DEP_HDP_PROBLEM_2OBJECTIVE:
        dep_hdp_problem_2o = HDP_MultiObjectiveVehicleRoutingProblem(
            ndp_customer_list=ndp_problem.get_ndp_customer_list(),
            number_of_hdp_customer=NUMBER_OF_HDP_CUSTOMER,
            range_of_hdp_customer=RANGE_OF_HDP_CUSTOMER,
            ndp_encoded_solutions=ndp_solution_handler.get_best_encoded_solutions(),
            optimize_similality=False,
        )

        # dep_hdp_problem_2o.visualize()

        dep_hdp_algorithm_2o = NSGA2(
            pop_size=len(ndp_solution_handler.get_best_encoded_solutions()),
            n_offsprings=20,
            sampling=CustomRandomSampling(NUMBER_OF_HDP_CUSTOMER),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        # Run the optimization
        dep_hdp_res_2o = minimize(
            dep_hdp_problem_2o, dep_hdp_algorithm_2o, ("n_gen", 200), verbose=False
        )

        # Create solution handler
        dep_hdp_solution_handler_2o = SolutionHandler(
            dep_hdp_problem_2o.get_map_graph()
        )
        dep_hdp_solution_handler_2o.set_result(dep_hdp_res_2o)
        dep_hdp_solution_handler_2o.print_best_decoded_solutions(1)

        if ENABLE_SOLUTION_VISUALIZATION:
            dep_hdp_solution_handler_2o.visualize_solution(
                "Dependent HDP problem - 2 objectives"
            )

        dep_hdp_solution_handler_2o.print_similarity(
            encoded_hdp_solution=dep_hdp_solution_handler_2o.get_best_encoded_solutions(
                1
            )[0],
            encoded_ndp_solution_list=ndp_solution_handler.get_best_encoded_solutions(),
        )

    if ENABLE_DEP_HDP_PROBLEM_3OBJECTIVE:
        dep_hdp_problem_3o = HDP_MultiObjectiveVehicleRoutingProblem(
            ndp_customer_list=ndp_problem.get_ndp_customer_list(),
            number_of_hdp_customer=NUMBER_OF_HDP_CUSTOMER,
            range_of_hdp_customer=RANGE_OF_HDP_CUSTOMER,
            ndp_encoded_solutions=ndp_solution_handler.get_best_encoded_solutions(),
            optimize_similality=True,
        )

        # dep_hdp_problem_3o.visualize()

        dep_hdp_algorithm_3o = NSGA2(
            pop_size=len(ndp_solution_handler.get_best_encoded_solutions()),
            n_offsprings=20,
            sampling=CustomRandomSampling(NUMBER_OF_HDP_CUSTOMER),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        # Run the optimization
        dep_hdp_res_3o = minimize(
            dep_hdp_problem_3o, dep_hdp_algorithm_3o, ("n_gen", 200), verbose=False
        )

        # Create solution handler
        dep_hdp_solution_handler_3o = SolutionHandler(
            dep_hdp_problem_3o.get_map_graph()
        )
        dep_hdp_solution_handler_3o.set_result(dep_hdp_res_3o)
        dep_hdp_solution_handler_3o.print_best_decoded_solutions(1)

        if ENABLE_SOLUTION_VISUALIZATION:
            dep_hdp_solution_handler_3o.visualize_solution(
                "Dependent HDP problem - 3 objectives"
            )

        dep_hdp_solution_handler_3o.print_similarity(
            encoded_hdp_solution=dep_hdp_solution_handler_3o.get_best_encoded_solutions(
                1
            )[0],
            encoded_ndp_solution_list=ndp_solution_handler.get_best_encoded_solutions(),
        )

    if ENABLE_DEP_HDP_PROBLEM_2OBJECTIVE_MEMETIC:
        dep_hdp_problem_2o_memetic = HDP_MultiObjectiveVehicleRoutingProblem(
            ndp_customer_list=ndp_problem.get_ndp_customer_list(),
            number_of_hdp_customer=NUMBER_OF_HDP_CUSTOMER,
            range_of_hdp_customer=RANGE_OF_HDP_CUSTOMER,
            ndp_encoded_solutions=ndp_solution_handler.get_best_encoded_solutions(),
            optimize_similality=False,
        )

        # dep_hdp_problem_2o_memetic.visualize()

        dep_hdp_algorithm_2o_memetic = NSGA2(
            pop_size=len(ndp_solution_handler.get_best_encoded_solutions()),
            n_offsprings=20,
            sampling=CustomRandomSampling(NUMBER_OF_HDP_CUSTOMER),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        # Run the optimization
        dep_hdp_res_2o_memetic = minimize(
            dep_hdp_problem_2o_memetic,
            dep_hdp_algorithm_2o_memetic,
            ("n_gen", 200),
            verbose=False,
        )

        # Create local search handler
        dep_hdp_local_search_handler_2o_memetic = LocalSearchHandler(
            dep_hdp_problem_2o_memetic.get_map_graph()
        )
        dep_hdp_local_search_handler_2o_memetic.set_result(dep_hdp_res_2o_memetic)
        dep_hdp_local_search_handler_2o_memetic.calculate_local_search(
            ndp_encoded_solution_list=ndp_solution_handler.get_best_encoded_solutions()
        )

        dep_hdp_local_search_handler_2o_memetic.print_best_decoded_solutions(1)

        if ENABLE_SOLUTION_VISUALIZATION:
            dep_hdp_local_search_handler_2o_memetic.visualize_solution(
                "Dependent HDP problem - 2 objectives - memetic"
            )

        dep_hdp_local_search_handler_2o_memetic.print_similarity(
            encoded_hdp_solution=dep_hdp_local_search_handler_2o_memetic.get_best_encoded_solutions(
                1
            )[
                0
            ],
            encoded_ndp_solution_list=ndp_solution_handler.get_best_encoded_solutions(),
        )


if __name__ == "__main__":
    main()
