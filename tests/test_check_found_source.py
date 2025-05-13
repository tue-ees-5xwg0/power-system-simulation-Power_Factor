import pytest

from power_system_simulation.assignment_1 import IDNotFoundError, check_found_source


def test_check_found_source():
    vertex_ids = [0, 2, 4, 6, 10]
    source_vertex_id = 3
    with pytest.raises(IDNotFoundError) as excinfo:
        check_found_source(source_vertex_id, vertex_ids)
    assert str(excinfo.value) == "Source vertex id not found"


# test_check_found_source()

# vertex_ids = [0, 2, 4, 6, 10]
# source=2

# check_source(source,vertex_ids)

# vertex_ids = [0, 2, 4, 6, 10]
# edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]

# check_pairs(edge_vertex_id_pairs,vertex_ids)
