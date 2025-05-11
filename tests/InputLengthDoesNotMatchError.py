def check_enabled(edge_enabled, edge_ids):
    if len(edge_enabled) != len(edge_ids):
        raise Exception("InputLengthDoesNotMatchError")


def check_pairs(edge_vertex_id_pairs, edge_ids):
    if len(edge_vertex_id_pairs) != len(edge_ids):
        raise Exception("InputLengthDoesNotMatchError")


# edge_ids = [1, 3, 5, 7, 8, 9]
# edge_enabled = [True, True, True, False, False, True]
# edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]

# check_enabled(edge_enabled,edge_ids)
# check_pairs(edge_vertex_id_pairs,edge_ids)
