from collections import defaultdict
import heapq


class WeightedUndirectedGraph(object):

    def __init__(self, vertices: int):
        self.V = vertices  # No. of vertices
        self.numberOfEdges = 0
        self.graph = defaultdict(list)  # default dictionary to store graph

    def addEdge(self, u: int, v: int, weight: int):
        """
        Function to add an edge to graph.

        Args:
            u (int): starting vertex
            v (int): finishing vertex
            weight (int): positive number value indicating weight
        """
        assert weight > 0, "Weight value must be positive!"

        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))
        self.numberOfEdges += 1

    def isConnected(self):
        self.visited = [False] * self.V

        # Find a vertex with non-zero degree
        i = 0  # Start from 0 for 0-based indexing
        for i in range(1, self.V):
            if len(self.graph[i]) != 0:
                break

        # If there are no edges in the graph, return false
        if i == self.V:
            return False

        # Start travel from a vertex with non-zero degree
        self.DFS(i, self.visited)

        # Check if all non-zero degree vertices are visited
        for i in range(1, self.V):
            if self.visited[i] == False and len(self.graph[i]) > 0:
                return False

        return True

    # Travel to all non-zero degree vertices
    def DFS(self, vertex: int, visited: list):
        """
        Depth First Search

        Args:
            vertex (int): current vertex
            visited (list): visited list
        """
        # Mark the current vertex as visited
        visited[vertex] = True

        # Recur for all the vertices adjacent to this vertex
        for i in [tupleElement[0] for tupleElement in self.graph[vertex]]:
            if visited[i] == False:
                self.DFS(i, visited)

    def findShortestPathDijkstra(self, startVertex: int, endVertex: int):
        # Check whether the path from startVertex to endVertex is existing
        self.isConnected()
        if not (self.visited[startVertex] and self.visited[endVertex]):
            return (float("infinity"), [])

        # Setup queue
        priorityQueue = [(0, startVertex)]
        distances = {vertex: float("infinity") for vertex in self.graph}
        previousVertex = {vertex: None for vertex in self.graph}
        distances[startVertex] = 0

        while priorityQueue:
            # Get the vertex with the smallest distance
            currentDistance, currentVertex = heapq.heappop(priorityQueue)

            # If we have reached the end vertex, we can reconstruct the path
            if currentVertex == endVertex:
                path = []
                while previousVertex[currentVertex] is not None:
                    path.insert(0, currentVertex)
                    currentVertex = previousVertex[currentVertex]
                path.insert(0, startVertex)
                return distances[endVertex], path

            # If a shorter path to the current vertex is found, continue
            if currentDistance > distances[currentVertex]:
                continue

            # Explore the neighbors of the current vertex
            for neighbor, weight in self.graph[currentVertex]:
                distance = currentDistance + weight

                # If a shorter path to the neighbor is found
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previousVertex[neighbor] = currentVertex
                    heapq.heappush(priorityQueue, (distance, neighbor))

        return (float("infinity"), [])


def main():
    # g1 = WeightedUndirectedGraph(20)
    # g1.addEdge(0, 1, 50)
    # g1.addEdge(0, 2, 55)
    # g1.addEdge(0, 3, 45)
    # g1.addEdge(0, 4, 30)
    # g1.addEdge(1, 2, 60)
    # g1.addEdge(1, 5, 62)
    # g1.addEdge(1, 6, 127)
    # g1.addEdge(2, 3, 43)
    # g1.addEdge(2, 6, 250)
    # g1.addEdge(2, 7, 35)
    # g1.addEdge(2, 8, 170)
    # g1.addEdge(3, 4, 90)
    # g1.addEdge(3, 8, 85)
    # g1.addEdge(3, 9, 40)
    # g1.addEdge(3, 10, 15)
    # g1.addEdge(4, 10, 230)
    # g1.addEdge(5, 6, 25)
    # g1.addEdge(5, 11, 136)
    # g1.addEdge(6, 7, 32)
    # g1.addEdge(6, 8, 30)
    # g1.addEdge(6, 12, 220)
    # g1.addEdge(7, 8, 120)
    # g1.addEdge(7, 11, 61)
    # g1.addEdge(7, 12, 88)
    # g1.addEdge(7, 13, 20)
    # g1.addEdge(8, 9, 61)
    # g1.addEdge(8, 13, 12)
    # g1.addEdge(8, 14, 23)
    # g1.addEdge(9, 10, 32)
    # g1.addEdge(9, 14, 194)
    # g1.addEdge(9, 15, 147)
    # g1.addEdge(9, 18, 110)
    # g1.addEdge(10, 15, 130)
    # g1.addEdge(11, 12, 144)
    # g1.addEdge(11, 16, 161)
    # g1.addEdge(12, 13, 24)
    # g1.addEdge(12, 16, 71)
    # g1.addEdge(12, 17, 92)
    # g1.addEdge(13, 14, 40)
    # g1.addEdge(13, 17, 134)
    # g1.addEdge(13, 19, 156)
    # g1.addEdge(14, 15, 77)
    # g1.addEdge(14, 18, 16)
    # g1.addEdge(14, 19, 17)
    # g1.addEdge(15, 18, 89)
    # g1.addEdge(16, 17, 26)
    # g1.addEdge(17, 19, 16)
    # g1.addEdge(18, 19, 72)
    # print(g1.findShortestPathDijkstra(0, 19))
    # print(g1.findShortestPathDijkstra(2, 11))
    # print(g1.findShortestPathDijkstra(16, 4))
    # print(g1.findShortestPathDijkstra(18, 1))

    g2 = WeightedUndirectedGraph(9)
    g2.addEdge(0, 1, 4)
    g2.addEdge(0, 7, 8)
    g2.addEdge(1, 2, 8)
    g2.addEdge(1, 7, 11)
    g2.addEdge(2, 3, 7)
    g2.addEdge(2, 5, 4)
    g2.addEdge(2, 8, 2)
    g2.addEdge(3, 4, 9)
    g2.addEdge(3, 5, 14)
    g2.addEdge(4, 5, 10)
    g2.addEdge(5, 6, 2)
    g2.addEdge(6, 7, 1)
    g2.addEdge(6, 8, 6)
    g2.addEdge(7, 8, 7)
    print(g2.findShortestPathDijkstra(0, 2))
    print(g2.findShortestPathDijkstra(0, 4))
    print(g2.findShortestPathDijkstra(0, 8))


if __name__ == "__main__":
    main()
