import pytest

import power_system_simulation.assignment_1 as a1


def test_check_connect():
    edge_ids = [1, 3, 5, 7, 8, 9]
    vertex_ids = [0, 2, 4, 6, 10]
    edge_enabled = [True, True, True, False, False, True]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(a1.GraphNotFullyConnectedError) as excinfo:
        a1.check_connect(vertex_ids, edge_ids, edge_enabled, edge_vertex_id_pairs)
    assert str(excinfo.value) == "Graph contains more than 1 component"


def test_check_cycle():
    edge_enabled = [True, True, True, False, True, True]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(a1.GraphCycleError) as excinfo:
        a1.check_cycle(edge_enabled, edge_vertex_id_pairs)
    assert str(excinfo.value) == "Graph contains a cycle"


def test_check_found_pairs():
    vertex_ids = [0, 3, 4, 6, 10]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(a1.IDNotFoundError) as excinfo:
        a1.check_found_pairs(edge_vertex_id_pairs, vertex_ids)
    assert str(excinfo.value) == "Vertex id not found in edge array"


def test_check_found_source():
    vertex_ids = [0, 2, 4, 6, 10]
    source_vertex_id = 3
    with pytest.raises(a1.IDNotFoundError) as excinfo:
        a1.check_found_source(source_vertex_id, vertex_ids)
    assert str(excinfo.value) == "Source vertex id not found"


def test_check_length_enabled():
    edge_enabled = [True, True, True, False, False, True]
    edge_ids = [1, 3, 7, 8, 9]
    with pytest.raises(a1.InputLengthDoesNotMatchError) as excinfo:
        a1.check_length_enabled(edge_enabled, edge_ids)
    assert str(excinfo.value) == "Number of enabled and disabled edges does not match number of total edges"


def test_check_length_pairs():
    edge_ids = [1, 3, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(a1.InputLengthDoesNotMatchError) as excinfo:
        a1.check_length_pairs(edge_vertex_id_pairs, edge_ids)
    assert str(excinfo.value) == "Number of vertex pairs does not match number of edges"


def test_check_unique():
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [3, 3, 5, 7, 8, 9]
    with pytest.raises(a1.IDNotUniqueError) as excinfo:
        a1.check_unique(vertex_ids, edge_ids)
    assert str(excinfo.value) == "Vertex or edge ids are not unique"
