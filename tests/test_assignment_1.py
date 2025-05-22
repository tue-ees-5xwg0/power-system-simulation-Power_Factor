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


def test_check_found_edges():
    disabled_edge_id = 2
    all_edges = [1, 3, 5, 7, 8, 9]
    with pytest.raises(a1.IDNotFoundError) as excinfo:
        a1.check_found_edges(disabled_edge_id, all_edges)
    assert str(excinfo.value) == "Disabled edge id not found in edge array"


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


def test_check_disabled():
    disabled_edge_id = 8
    edge_ids = [1, 3, 5, 8, 9]
    edge_enabled = [True, True, True, False, False]
    with pytest.raises(a1.EdgeAlreadyDisabledError) as excinfo:
        a1.check_disabled(disabled_edge_id, edge_ids, edge_enabled)
    assert str(excinfo.value) == "Edge is already disabled"


def test_check_downstream_vertex():
    vertex_ids = [0, 2, 4, 6, 8, 10, 12]
    edge_ids = [1, 3, 5, 7, 9, 11]
    edge_vertex_id_pairs = [(0, 2), (2, 4), (2, 6), (4, 8), (8, 10), (6, 12)]
    edge_enabled = [True, True, True, True, True, True]
    source_vertex_id = 0

    """
    vertex_0 (source) --edge_1-- vertex_2 --edge_3-- vertex_4--edge 7--vertex 8 --edge 9--vertex 10
                                    |
                                  edge 5
                                    |
                                 vertex 6  --edge 11 --vertex 12
    """

    test = a1.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    assert test.find_downstream_vertices(1) == [2, 4, 6, 8, 10, 12]

    with pytest.raises(a1.IDNotFoundError) as excinfo:
        test.find_downstream_vertices(2)
    assert str(excinfo.value) == "Edge ID not found."


# test_check_downstream_vertex()


def test_find_alternative_edges():
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 9, 8]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    edge_enabled = [True, True, True, False, True, False]
    source_vertex_id = 0

    """
            vertex_0 (source) --edge_1(enabled)-- vertex_2 --edge_9(enabled)-- vertex_10
                 |                               |
                 |                           edge_7(disabled)
                 |                               |
                 -----------edge_3(enabled)-- vertex_4
                 |                               |
                 |                           edge_8(disabled)
                 |                               |
                 -----------edge_5(enabled)-- vertex_6
    """

    test = a1.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    assert test.find_alternative_edges(1) == [7]
    assert test.find_alternative_edges(3) == [7, 8]
    assert test.find_alternative_edges(5) == [8]
    assert test.find_alternative_edges(9) == []

    with pytest.raises(a1.IDNotFoundError) as excinfo:
        test.find_alternative_edges(6)
    assert str(excinfo.value) == "Disabled edge id not found in edge array"

    with pytest.raises(a1.EdgeAlreadyDisabledError) as excinfo:
        test.find_alternative_edges(7)
    assert str(excinfo.value) == "Edge is already disabled"


# test_find_alternative_edges()
