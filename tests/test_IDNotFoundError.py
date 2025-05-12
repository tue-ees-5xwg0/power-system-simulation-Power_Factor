from IDNotFoundError import check_source, check_pairs

def test_source():
    """Check if adding function works"""
    vertex_ids = [0, 2, 4, 6, 10]
    source_vertex_id = 1

    try:
        check_source(source_vertex_id, vertex_ids)
    except Exception as e:
        assert str(e) == "IDNotFoundError"
