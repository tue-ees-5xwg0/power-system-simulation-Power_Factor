import pytest

from power_system_simulation.assignment_1 import GraphNotFullyConnectedError, check_connect


def test_check_connect():
    edge_ids = [1, 3, 5, 7, 8, 9]
    vertex_ids = [0, 2, 4, 6, 10]
    edge_enabled = [True, True, True, False, False, True]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(GraphNotFullyConnectedError) as excinfo:
        check_connect(vertex_ids, edge_ids, edge_enabled, edge_vertex_id_pairs)
    assert str(excinfo.value) == "Graph contains more than 1 component"


# test_check_connect()
# edge_ids = [1, 3, 5, 7, 8, 9]
# vertex_ids = [0, 2, 4, 6, 10]
# edge_enabled = [True, True, True, False, False, True]
# edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
# check(vertex_ids, edge_ids,edge_enabled,edge_vertex_id_pairs)
