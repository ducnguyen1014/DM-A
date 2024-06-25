# Python program to check if a given graph is Eulerian or not
# Complexity : O(V+E)

# This class represents a undirected graph using adjacency list representation

from collections import defaultdict


class Graph(object):

    def __init__(self, vertices: int):
        self.V = vertices  # No. of vertices
        self.numberOfEdges = 0
        self.graph = defaultdict(list)  # default dictionary to store graph

    # function to add an edge to graph
    def addEdge(self, u: int, v: int):
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.numberOfEdges += 1

    def isConnected(self):
        visited = [False] * self.V

        # Find a vertex with non-zero degree
        i = 0
        for i in range(self.V):
            if len(self.graph[i]) != 0:
                break

        # If there are no edges in the graph, return false
        if i == self.V - 1:
            return False

        # Start travel from a vertex with non-zero degree
        self.travelToAllVertices(i, visited)

        # Check if all non-zero degree vertices are visited
        for i in range(self.V):
            if visited[i] == False and len(self.graph[i]) > 0:
                return False

        return True

    # Travel to all non-zero degree vertices
    def travelToAllVertices(self, vertex: int, visited: list):
        # Mark the current node as visited
        visited[vertex] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[vertex]:
            if visited[i] == False:
                self.travelToAllVertices(i, visited)

    """The function returns one of the following values
    0 --> If graph is not Eulerian
    1 --> If graph has an Euler path (Semi-Eulerian)
    2 --> If graph has an Euler Circuit (Eulerian) """

    def isEulerian(self):
        # 1. Check if all non-zero degree vertices are connected
        # We don’t care about vertices with zero degree because
        # they don’t belong to Eulerian Cycle or Path (we only consider all edges).

        if self.isConnected() == False:
            return 0
        else:
            # Count vertices with odd degree
            odd = 0
            for i in range(self.V):
                if len(self.graph[i]) % 2 != 0:
                    odd += 1

            """
            If odd count is 0, then Eulerian
            If odd count is 2, then semi-Eulerian.
            If count is more than 2, then graph is not Eulerian
            Note that odd count can never be 1 for undirected graph
            """

            if odd == 0:
                return 2
            elif odd == 2:
                return 1
            elif odd > 2:
                return 0

    def travelThroughEdges(
        self, startVertex: int, currentVertex: int, traveledEdges: list, path: list
    ):
        for nextVertex in self.graph[currentVertex]:
            # Have not traveled
            if (
                nextVertex != startVertex
                and {currentVertex, nextVertex} not in traveledEdges
            ):
                traveledEdges.append({currentVertex, nextVertex})
                path.append(nextVertex)
                path = self.travelThroughEdges(
                    startVertex, nextVertex, traveledEdges, path
                )

            # In case come back to a node that has at least 4 degrees
            elif nextVertex == startVertex and len(path) == self.numberOfEdges:
                path.append(startVertex)
                traveledEdges.append({currentVertex, nextVertex})

            # Come back to started node
            else:
                continue

        return path

    def findEulerianCycle(self):
        # Find the Eulerian cycle

        if self.isEulerian() == 2:
            i = 0
            while len(self.graph[i]) == 0:
                i += 1

            return self.travelThroughEdges(i, i, [], [i])

        # Find the Eulerian path
        elif self.isEulerian() == 1:
            i = 0
            while len(self.graph[i]) % 2 != 1:
                i += 1

            return self.travelThroughEdges(i, i, [], [i])

        # Neither the Eulerian cycle nor the Eulerian path
        else:
            return []


def main():
    # g1 = Graph(5)
    # g1.addEdge(1, 2)
    # g1.addEdge(2, 0)
    # g1.addEdge(0, 3)
    # g1.addEdge(3, 4)
    # print(g1.isEulerian())
    # print(g1.findEulerianCycle())

    g2 = Graph(11)
    g2.addEdge(2, 1)  # 1
    g2.addEdge(1, 0)  # 2
    g2.addEdge(0, 10)  # 3
    g2.addEdge(10, 9)  # 4
    g2.addEdge(9, 8)  # 5
    g2.addEdge(8, 5)  # 6
    g2.addEdge(5, 4)  # 7
    g2.addEdge(4, 3)  # 8
    g2.addEdge(3, 2)  # 9
    g2.addEdge(1, 3)  # 10
    g2.addEdge(1, 6)  # 11
    g2.addEdge(6, 3)  # 12
    g2.addEdge(6, 7)  # 13
    g2.addEdge(6, 5)  # 14
    g2.addEdge(7, 10)  # 15
    g2.addEdge(7, 8)  # 16
    g2.addEdge(7, 5)  # 17
    g2.addEdge(10, 8)  # 18
    print(g2.isEulerian())
    print(g2.findEulerianCycle())


if __name__ == "__main__":
    main()
