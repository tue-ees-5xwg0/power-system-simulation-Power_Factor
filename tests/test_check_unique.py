import pytest
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from power_system_simulation.assignment_1 import IDNotUniqueError, check_unique


def test_check_unique():
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [3, 3, 5, 7, 8, 9]
    with pytest.raises(IDNotUniqueError) as excinfo:
        check_unique(vertex_ids, edge_ids)
    assert str(excinfo.value) == "Vertex or edge ids are not unique"


# vertex_ids = [0, 2, 4, 6, 10]
# edge_ids = [1, 3, 5, 7, 8, 9]

# test_check_unique()
