from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

"""
vertex_ids: list of vertex ids
edge_ids: list of edge ids
edge_vertex_id_pairs: list of tuples of two integer
edge_enabled: list of bools indicating of an edge is enabled or not
source_vertex_id: vertex id of the source in the graph
"""


class GraphProcessor:
    def __init__(
        self,
        vertex_ids: List[int],
        edge_ids: List[int],
        edge_vertex_id_pairs: List[Tuple[int, int]],
        edge_enabled: List[bool],
        source_vertex_id: int,
    ) -> None:
        self.vertex_ids = vertex_ids
        self.edge_ids = edge_ids
        self.edge_vertex_id_pairs = edge_vertex_id_pairs
        self.edge_enabled = edge_enabled
        self.source_vertex_id = source_vertex_id


vertex_ids = [0, 2, 4, 6, 10]
edge_ids = [1, 3, 5, 7, 8, 9]
edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
edge_enabled = [True, True, True, False, False, True]
source_vertex_id = 0

g = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
