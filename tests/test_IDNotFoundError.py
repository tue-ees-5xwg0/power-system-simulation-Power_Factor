from power_system_simulation.IDNotFoundError import check_source, check_pairs
def test_source():
    """Check if function works"""
    vertex_ids = [0, 2, 4, 6, 10]
    source_vertex_id = 1

    assert check_source(source_vertex_id, vertex_ids), "IDNotFoundError"
    # assert check_source(0, vertex_ids)

def test_pairs():
    """Check if adding function works"""
    vertex_ids = [0, 2, 4, 6, 11]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
    assert check_pairs(edge_vertex_id_pairs, vertex_ids), "IDNotFoundError"


    vertex_ids = [0, 2, 4, 6, 10]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 7), (2, 4), (2, 10), (4, 6)]
    assert check_pairs(edge_vertex_id_pairs, vertex_ids), "IDNotFoundError"

