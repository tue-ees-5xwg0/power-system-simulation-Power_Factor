import pytest

from power_system_simulation.assignment_1 import InputLengthDoesNotMatchError, check_length_pairs


def test_check_length_pairs():
    edge_ids = [1, 3, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(InputLengthDoesNotMatchError) as excinfo:
        check_length_pairs(edge_vertex_id_pairs, edge_ids)
    assert str(excinfo.value) == "Number of vertex pairs does not match number of edges"


# test_check_length_pairs()
