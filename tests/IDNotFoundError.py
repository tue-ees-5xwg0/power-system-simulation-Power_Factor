def check_source(source_vertex_id,vertex_ids):
    if source_vertex_id not in vertex_ids:
        raise Exception("Source vertex ID not found")
    
def check_pairs(edge_vertex_id_pairs,vertex_ids):
    if all(all(elem in vertex_ids for elem in t) for t in edge_vertex_id_pairs)==False:
        raise Exception("Edge vertex pairs must contain valid vertex IDs")
#vertex_ids = [0, 2, 4, 6, 10]
#source=2

#check_source(source,vertex_ids)

#vertex_ids = [0, 2, 4, 6, 10]
#edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 10), (4, 6)]

#check_pairs(edge_vertex_id_pairs,vertex_ids)
