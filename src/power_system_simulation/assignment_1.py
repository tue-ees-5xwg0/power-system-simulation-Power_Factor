"""
This is a skeleton for the graph processing assignment.

We define a graph processor class with some function skeletons.
"""

from typing import List, Tuple

import networkx as nx
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components


class IDNotFoundError(Exception):
    """Raised when a given vertex or edge ID is not found in the graph."""


class InputLengthDoesNotMatchError(Exception):
    """Raised when input lists have mismatched lengths."""


class IDNotUniqueError(Exception):
    """Raised when duplicate IDs are found in vertex_ids or edge_ids."""


class GraphNotFullyConnectedError(Exception):
    """Raised when the graph is not fully connected."""


class GraphCycleError(Exception):
    """Raised when the graph contains cycles."""


class EdgeAlreadyDisabledError(Exception):
    """Raised when attempting to disable an already disabled edge."""


def check_cycle(edge_enabled, edge_vertex_id_pairs):
    G = nx.Graph()
    for (u, v), enabled in zip(edge_vertex_id_pairs, edge_enabled):
        if enabled:
            G.add_edge(u, v)

    has_cycle = nx.is_forest(G)  # A forest is a graph with no undirected cycles
    if has_cycle == False:
        raise GraphCycleError("Graph contains a cycle")


def check_connect(vertex_ids, edge_ids, edge_enabled, edge_vertex_id_pairs):
    size = len(vertex_ids)
    sparseMatrix = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            if ((vertex_ids[i], vertex_ids[j]) in edge_vertex_id_pairs) and sparseMatrix[i][j] == 0:
                if edge_enabled[edge_vertex_id_pairs.index((vertex_ids[i], vertex_ids[j]))]:
                    sparseMatrix[i][j] = vertex_ids[j]
                    sparseMatrix[j][i] = vertex_ids[i]
            elif ((vertex_ids[j], vertex_ids[i]) in edge_vertex_id_pairs) and sparseMatrix[i][j] == 0:
                if edge_enabled[edge_vertex_id_pairs.index((vertex_ids[j], vertex_ids[i]))]:
                    sparseMatrix[i][j] = vertex_ids[j]
                    sparseMatrix[j][i] = vertex_ids[i]
    graph = csr_array(sparseMatrix)
    components = connected_components(graph)
    if components[0] > 1:
        raise GraphNotFullyConnectedError("Graph contains more than 1 component")


def check_found_source(source_vertex_id, vertex_ids):
    if source_vertex_id not in vertex_ids:
        raise IDNotFoundError("Source vertex id not found")


def check_found_pairs(edge_vertex_id_pairs, vertex_ids):
    if all(all(elem in vertex_ids for elem in t) for t in edge_vertex_id_pairs) == False:
        raise IDNotFoundError("Vertex id not found in edge array")


def check_length_enabled(edge_enabled, edge_ids):
    if len(edge_enabled) != len(edge_ids):
        raise InputLengthDoesNotMatchError("Number of enabled and disabled edges does not match number of total edges")


def check_length_pairs(edge_vertex_id_pairs, edge_ids):
    if len(edge_vertex_id_pairs) != len(edge_ids):
        raise InputLengthDoesNotMatchError("Number of vertex pairs does not match number of edges")


def check_unique(vertex_ids, edge_ids):
    ver = list(set(vertex_ids))
    edge = list(set(edge_ids))
    if len(ver) != len(vertex_ids) or len(edge) != len(edge_ids):
        raise IDNotUniqueError("Vertex or edge ids are not unique")


class GraphProcessor(nx.Graph):
    """
    General documentation of this class.
    You need to describe the purpose of this class and the functions in it.
    We are using an undirected graph in the processor.
    """

    def __init__(
        self,
        vertex_ids: List[int],
        edge_ids: List[int],
        edge_vertex_id_pairs: List[Tuple[int, int]],
        edge_enabled: List[bool],
        source_vertex_id: int,
    ) -> None:
        """
        Initialize a graph processor object with an undirected graph.
        Only the edges which are enabled are taken into account.
        Check if the input is valid and raise exceptions if not.
        The following conditions should be checked:
            1. vertex_ids and edge_ids should be unique. (IDNotUniqueError)
            2. edge_vertex_id_pairs should have the same length as edge_ids. (InputLengthDoesNotMatchError)
            3. edge_vertex_id_pairs should contain valid vertex ids. (IDNotFoundError)
            4. edge_enabled should have the same length as edge_ids. (InputLengthDoesNotMatchError)
            5. source_vertex_id should be a valid vertex id. (IDNotFoundError)
            6. The graph should be fully connected. (GraphNotFullyConnectedError)
            7. The graph should not contain cycles. (GraphCycleError)
        If one certain condition is not satisfied, the error in the parentheses should be raised.

        Args:
            vertex_ids: list of vertex ids
            edge_ids: liest of edge ids
            edge_vertex_id_pairs: list of tuples of two integer
                Each tuple is a vertex id pair of the edge.
            edge_enabled: list of bools indicating of an edge is enabled or not
            source_vertex_id: vertex id of the source in the graph
        """
        check_unique(vertex_ids, edge_ids)
        check_length_pairs(edge_vertex_id_pairs, edge_ids)
        check_found_pairs(edge_vertex_id_pairs, vertex_ids)
        check_length_enabled(edge_enabled, edge_ids)
        check_found_source(source_vertex_id, vertex_ids)
        check_connect(vertex_ids, edge_ids, edge_enabled, edge_vertex_id_pairs)
        check_cycle(edge_enabled, edge_vertex_id_pairs)

        nx.Graph.__init__(self)
        self.source_vertex_id = source_vertex_id
        self.add_nodes_from(vertex_ids)
        for i, (u, v) in enumerate(edge_vertex_id_pairs):
            self.add_edge(u, v, id=edge_ids[i], enabled=edge_enabled[i])

        # vertex_ids = [0, 2, 4, 6, 10]

    # edge_ids = [1, 3, 5, 7, 8, 9]
    # edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    # edge_enabled = [True, True, True, False, True, True]
    # source_vertex_id = 0

    # g = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)

    def find_downstream_vertices(self, edge_id: int) -> List[int]:
        """
        Given an edge id, return all the vertices which are in the downstream of the edge,
            with respect to the source vertex.
            Including the downstream vertex of the edge itself!

        Only enabled edges should be taken into account in the analysis.
        If the given edge_id is a disabled edge, it should return empty list.
        If the given edge_id does not exist, it should raise IDNotFoundError.


        For example, given the following graph (all edges enabled):

            vertex_0 (source) --edge_1-- vertex_2 --edge_3-- vertex_4

        Call find_downstream_vertices with edge_id=1 will return [2, 4]
        Call find_downstream_vertices with edge_id=3 will return [4]

        Args:
            edge_id: edge id to be searched

        Returns:
            A list of all downstream vertices.
        """

        self._edge_index = {eid: i for i, eid in enumerate(edge_ids)}
        if edge_id not in self._edge_index:
            raise IDNotFoundError("Edge id not found")
        
        idx = 0
        for i in range(len(edge_ids)):
            if edge_ids[i] == edge_id:
                idx = i
                if edge_enabled[i] == False:
                    return []
                else:
                    break
        print(edge_vertex_id_pairs[idx][1])
        dfs_edges = list(nx.dfs_successors(self, source=edge_vertex_id_pairs[idx][1]))
        print("DFS Edges:", dfs_edges)
        pass

    def find_alternative_edges(self, disabled_edge_id: int) -> List[int]:
        """
        Given an enabled edge, do the following analysis:
            If the edge is going to be disabled,
                which (currently disabled) edge can be enabled to ensure
                that the graph is again fully connected and acyclic?
            Return a list of all alternative edges.
        If the disabled_edge_id is not a valid edge id, it should raise IDNotFoundError.
        If the disabled_edge_id is already disabled, it should raise EdgeAlreadyDisabledError.
        If there are no alternative to make the graph fully connected again, it should return empty list.


        For example, given the following graph:

        vertex_0 (source) --edge_1(enabled)-- vertex_2 --edge_9(enabled)-- vertex_10
                 |                               |
                 |                           edge_7(disabled)
                 |                               |
                 -----------edge_3(enabled)-- vertex_4
                 |                               |
                 |                           edge_8(disabled)
                 |                               |
                 -----------edge_5(enabled)-- vertex_6

        Call find_alternative_edges with disabled_edge_id=1 will return [7]
        Call find_alternative_edges with disabled_edge_id=3 will return [7, 8]
        Call find_alternative_edges with disabled_edge_id=5 will return [8]
        Call find_alternative_edges with disabled_edge_id=9 will return []

        Args:
            disabled_edge_id: edge id (which is currently enabled) to be disabled

        Returns:
            A list of alternative edge ids.
        """
        # put your implementation here
        pass

vertex_ids = [0, 2, 4, 6, 10]
edge_ids = [1, 3, 5, 7]
edge_vertex_id_pairs = [(0, 2), (0, 4), (2, 6), (2, 10)]
edge_enabled = [True, True, True, True]
source_vertex_id = 0
g = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
g.find_downstream_vertices(1)