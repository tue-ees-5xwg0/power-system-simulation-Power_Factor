import pytest

from power_system_simulation.assignment_1 import IDNotFoundError, check_found_pairs


def test_check_found_pairs():
    vertex_ids = [0, 3, 4, 6, 10]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(IDNotFoundError) as excinfo:
        check_found_pairs(edge_vertex_id_pairs, vertex_ids)
    assert str(excinfo.value) == "Vertex id not found in edge array"


# vertex_ids = [0, 2, 4, 6, 10]
# edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
# test_check_found_pairs()
