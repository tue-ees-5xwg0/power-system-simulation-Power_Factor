import pytest

import power_system_simulation.assignment_1 as a1


def test_check_connect():
    '''Test that the graph is fully connected with no components missing.'''
    edge_ids = [1, 3, 5, 7, 8, 9]
    vertex_ids = [0, 2, 4, 6, 10]
    edge_enabled = [True, True, True, False, False, True]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(a1.GraphNotFullyConnectedError):
        a1.check_connect(vertex_ids, edge_ids, edge_enabled, edge_vertex_id_pairs)


def test_check_cycle():
    '''Test that the graph does not contain a cycle.'''
    edge_enabled = [True, True, True, False, True, True]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(a1.GraphCycleError):
        a1.check_cycle(edge_enabled, edge_vertex_id_pairs)


def test_check_found_pairs():
    '''Test that all vertex ids in edge_vertex_id_pairs are found in vertex_ids.'''
    vertex_ids = [0, 3, 4, 6, 10]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(a1.IDNotFoundError):
        a1.check_found_pairs(edge_vertex_id_pairs, vertex_ids)


def test_check_found_source():
    '''Test that the source vertex id is found in vertex_ids.'''
    vertex_ids = [0, 2, 4, 6, 10]
    source_vertex_id = 3
    with pytest.raises(a1.IDNotFoundError):
        a1.check_found_source(source_vertex_id, vertex_ids)


def test_check_found_edges():
    '''Test that the disabled edge id is found in all_edges.'''
    disabled_edge_id = 2
    all_edges = [1, 3, 5, 7, 8, 9]
    with pytest.raises(a1.IDNotFoundError):
        a1.check_found_edges(disabled_edge_id, all_edges)


def test_check_length_enabled():
    '''Test that the number of enabled and disabled edges matches the number of total edges.'''
    edge_enabled = [True, True, True, False, False, True]
    edge_ids = [1, 3, 7, 8, 9]
    with pytest.raises(a1.InputLengthDoesNotMatchError):
        a1.check_length_enabled(edge_enabled, edge_ids)


def test_check_length_pairs():
    '''Test that the number of vertex pairs matches the number of edges.'''
    edge_ids = [1, 3, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    with pytest.raises(a1.InputLengthDoesNotMatchError):
        a1.check_length_pairs(edge_vertex_id_pairs, edge_ids)


def test_check_unique():
    '''Test that vertex and edge ids are unique.'''
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [3, 3, 5, 7, 8, 9]
    with pytest.raises(a1.IDNotUniqueError):
        a1.check_unique(vertex_ids, edge_ids)


def test_check_disabled():
    '''Test that the disabled edge id is not already disabled.'''
    disabled_edge_id = 8
    edge_ids = [1, 3, 5, 8, 9]
    edge_enabled = [True, True, True, False, False]
    with pytest.raises(a1.EdgeAlreadyDisabledError):
        a1.check_disabled(disabled_edge_id, edge_ids, edge_enabled)


def test_check_downstream_vertex():
    '''Test that the downstream vertex id is found in vertex_ids.'''
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

    with pytest.raises(a1.IDNotFoundError):
        test.find_downstream_vertices(2)


# test_check_downstream_vertex()


def test_find_alternative_edges():
    '''Test that the alternative edges are found correctly.'''
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

    with pytest.raises(a1.IDNotFoundError):
        test.find_alternative_edges(6)

    with pytest.raises(a1.EdgeAlreadyDisabledError):
        test.find_alternative_edges(7)

test_check_found_source()