from power_system_simulation.IDNotUniqueError import check

def test_IDUnique():
    """Check if function works"""
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 8, 9]

    assert check(vertex_ids, edge_ids), "IDNotUniqueError"

test_IDUnique()