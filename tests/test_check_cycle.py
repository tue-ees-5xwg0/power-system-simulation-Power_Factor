import pytest

from power_system_simulation.assignment_1 import GraphCycleError, check_cycle


def test_check_cycle():
    edge_enabled = [True, True, True, False, True, True]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(GraphCycleError) as excinfo:
        check_cycle(edge_enabled, edge_vertex_id_pairs)
    assert str(excinfo.value) == "Graph contains a cycle"


# test_check_cycle()
# edge_enabled = [True, True, True, False, True, False]
# edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]

# check(edge_enabled,edge_vertex_id_pairs)
