def check(vertex_ids, edge_ids):
    ver = list(set(vertex_ids))
    edge = res = list(set(edge_ids))
    if len(ver) != len(vertex_ids) or len(edge) != len(edge_ids):
        raise Exception("Vertex and edge ids must be unique")


# vertex_ids = [0, 2, 4, 6, 10]
# edge_ids = [1, 3, 5, 7, 8, 9]

# check(vertex_ids,edge_ids)
