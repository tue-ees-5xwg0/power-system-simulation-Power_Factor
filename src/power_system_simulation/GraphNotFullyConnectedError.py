from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components


def check(vertex_ids, edge_ids, edge_enabled, edge_vertex_id_pairs):
    size = len(vertex_ids)
    sparseMatrix = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            if ((vertex_ids[i], vertex_ids[j]) in edge_vertex_id_pairs) and sparseMatrix[i][j] == 0:
                if edge_enabled[edge_vertex_id_pairs.index((vertex_ids[i], vertex_ids[j]))]:
                    sparseMatrix[i][j] = vertex_ids[j]
                    sparseMatrix[j][i] = vertex_ids[i]
            elif ((vertex_ids[j], vertex_ids[i]) in edge_vertex_id_pairs) and sparseMatrix[i][j] == 0:
                if edge_enabled[edge_vertex_id_pairs.index((vertex_ids[j], vertex_ids[i]))]:
                    sparseMatrix[i][j] = vertex_ids[j]
                    sparseMatrix[j][i] = vertex_ids[i]
    graph = csr_array(sparseMatrix)
    components = connected_components(graph)
    if components[0] > 1:
        raise Exception("GraphNotFullyConnectedError")


# edge_ids = [1, 3, 5, 7, 8, 9]
# vertex_ids = [0, 2, 4, 6, 10]
# edge_enabled = [True, True, True, False, False, True]
# edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]
# check(vertex_ids, edge_ids,edge_enabled,edge_vertex_id_pairs)
