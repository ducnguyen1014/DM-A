import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


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
            print("Road already exists")

    def visualize_coordinates(self):
        """
        Visualize the graph with coordinates as points and roads as lines.
        """
        plt.figure(figsize=(8, 6))

        # Plot each location and the roads between them
        for location, neighbors in self.graph.items():
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
        plt.grid(True)
        plt.show()


def main():
    loc1 = Coordinates(40.7128, -74.0060)  # New York City
    loc2 = Coordinates(34.0522, -118.2437)  # Los Angeles
    loc3 = Coordinates(51.5074, -0.1278)  # London

    # Create the map graph
    map_graph = MapGraph()

    # Add locations and roads
    map_graph.add_road(loc1, loc2)
    map_graph.add_road(loc1, loc2)

    # Visualize the graph
    map_graph.visualize_coordinates()


if __name__ == "__main__":
    main()
