import networkx as nx
def check(edge_enabled,edge_vertex_id_pairs):
    G = nx.Graph()
    for (u, v), enabled in zip(edge_vertex_id_pairs, edge_enabled):
        if enabled:
            G.add_edge(u, v)

    has_cycle = nx.is_forest(G)
    if (has_cycle==False):
        raise Exception("GraphCycleError")
    
#edge_enabled = [True, True, True, False, True, False]
#edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]

#check(edge_enabled,edge_vertex_id_pairs)
