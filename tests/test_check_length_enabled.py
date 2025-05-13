import pytest

from power_system_simulation.assignment_1 import InputLengthDoesNotMatchError, check_length_enabled


def test_check_length_enabled():
    edge_enabled = [True, True, True, False, False, True]
    edge_ids = [1, 3, 7, 8, 9]
    with pytest.raises(InputLengthDoesNotMatchError) as excinfo:
        check_length_enabled(edge_enabled, edge_ids)
    assert str(excinfo.value) == "Number of enabled and disabled edges does not match number of total edges"


# test_check_length_enabled()
# edge_ids = [1, 3, 5, 7, 8, 9]
# edge_enabled = [True, True, True, False, False, True]
# edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]

# check_enabled(edge_enabled,edge_ids)
# check_pairs(edge_vertex_id_pairs,edge_ids)
