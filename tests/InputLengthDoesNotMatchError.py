def check(edge_enabled, edge_ids):
    if len(edge_enabled) != len(edge_ids):
        raise Exception("The number of enabled edges should match the number of specified input edges")


# edge_ids = [1, 3, 5, 7, 8, 9]
# edge_enabled = [True, True, True, False, False, True]

# check(edge_enabled,edge_ids)
