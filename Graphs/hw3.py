from collections import defaultdict


class UndirectedGraph(object):

    def __init__(self, vertices: int):
        self.V = vertices  # No. of vertices
        self.numberOfEdges = 0
        self.graph = defaultdict(list)  # default dictionary to store graph

    def addEdge(self, u: int, v: int):
        """
        Function to add an edge to graph.

        Args:
            u (int): starting vertex
            v (int): finishing vertex
        """

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
        self.DFS(i, visited)

        # Check if all non-zero degree vertices are visited
        for i in range(self.V):
            if visited[i] == False and len(self.graph[i]) > 0:
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
        # Mark the current node as visited
        visited[vertex] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[vertex]:
            if visited[i] == False:
                self.DFS(i, visited)

    def isEulerian(self):
        """The function returns one of the following values

        0 --> If graph is not Eulerian

        1 --> If graph has an Euler path (Semi-Eulerian)

        2 --> If graph has an Euler circuit (Eulerian)

        Note that odd count can never be 1 for undirected graph.

        """

        # Check if all non-zero degree vertices are connected
        # We don’t care about vertices with zero degree because
        # they don’t belong to Eulerian Circuit or Path (we only consider all edges).

        if self.isConnected() == False:
            return 0
        else:
            # Count vertices with odd degree
            odd = 0
            for i in range(self.V):
                if len(self.graph[i]) % 2 != 0:
                    odd += 1

            if odd == 0:
                return 2
            elif odd == 2:
                return 1
            elif odd > 2:
                return 0

    def travelThroughEdges(self, currentVertex: int, traveledEdges: list, path: list):
        for nextVertex in self.graph[currentVertex]:

            # Have traveled to all edges in the graph
            if len(traveledEdges) == self.numberOfEdges:
                return path

            # Have not traveled through
            if {currentVertex, nextVertex} not in traveledEdges:
                traveledEdges.append({currentVertex, nextVertex})
                path.append(nextVertex)
                path = self.travelThroughEdges(nextVertex, traveledEdges, path)

                if len(traveledEdges) == self.numberOfEdges:
                    return path
                else:
                    traveledEdges.pop()
                    path.pop()

            # Have traveled through this vertex
            else:
                continue

        return path

    def makeEulerianPath(self):
        """
        Make Eulerian path of the graph.

        Eulerian circuit: a path that goes through each and every edge of the graph exactly once
        and finishes at the starting vertex.

        Eulerian path: a path that goes through each and every edge of the graph exactly once.

        Returns:
            tuple: (index, path)

            index = 2 => Eulerian circuit

            index = 1 => Eulerian path

            index = 0 => no Eulerian circuit / path
        """

        # Find the Eulerian circuit
        eulerianType = self.isEulerian()

        if eulerianType == 2:
            i = 0
            while len(self.graph[i]) == 0:
                i += 1

            return (2, self.travelThroughEdges(i, [], [i]))

        # Find the Eulerian path
        elif eulerianType == 1:
            i = 0
            while len(self.graph[i]) % 2 != 1:
                i += 1

            return (1, self.travelThroughEdges(i, [], [i]))

        # Create edges to get Eulerian path
        else:
            oddDegreeVertex = []
            for i in range(self.V):
                if len(self.graph[i]) % 2 == 1:
                    oddDegreeVertex.append(i)

            while len(oddDegreeVertex) > 2:
                self.addEdge(oddDegreeVertex[0], oddDegreeVertex[1])
                oddDegreeVertex = oddDegreeVertex[2:]

            startVertex = oddDegreeVertex[0]
            return (1, self.travelThroughEdges(startVertex, [], [startVertex]))


def main():
    g1 = UndirectedGraph(11)
    g1.addEdge(0, 1)  # 1
    g1.addEdge(0, 10)  # 2
    g1.addEdge(1, 2)  # 3
    g1.addEdge(1, 3)  # 4
    g1.addEdge(2, 3)  # 5
    g1.addEdge(3, 4)  # 6
    g1.addEdge(3, 6)  # 7
    g1.addEdge(4, 5)  # 8
    g1.addEdge(5, 6)  # 9
    g1.addEdge(6, 7)  # 10
    g1.addEdge(7, 8)  # 11
    g1.addEdge(7, 10)  # 12
    g1.addEdge(8, 9)  # 13
    g1.addEdge(8, 10)  # 14
    g1.addEdge(9, 10)  # 15
    print(g1.isEulerian())
    print(g1.makeEulerianPath())


if __name__ == "__main__":
    main()
