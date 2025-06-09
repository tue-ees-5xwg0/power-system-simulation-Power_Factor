from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import power_system_simulation.assignment_1 as a1
import power_system_simulation.assignment_3 as a3


def test_check_source_transformer():
    meta = {"lv_busbar": 1, "lv_feeders": [16, 20], "mv_source_node": 0, "source": 10, "transformer": 11}
    with pytest.raises(a3.MoreThanOneTransformerOrSource) as excinfo:
        a3.check_source_transformer(meta)
    assert str(excinfo.value) == "LV grid contains more than one source or transformer"


# test_check_source_transformer()


def test_check_valid_LV_ids():
    LV_ids = [15, 20]
    Line_ids = [16, 17, 18, 19, 20, 21, 22, 23, 24]
    with pytest.raises(a3.InvalidLVIds) as excinfo:
        a3.check_valid_LV_ids(LV_ids, Line_ids)
    assert str(excinfo.value) == "LV feeders contain invalid ids"


# test_check_valid_LV_ids()


def test_check_line_transformer_nodes():
    lines_from_nodes = [1, 1]
    transformer_to_node = 2
    with pytest.raises(a3.NonMatchingTransformerLineNodes) as excinfo:
        a3.check_line_transformer_nodes(lines_from_nodes, transformer_to_node)
    assert (
        str(excinfo.value)
        == "The lines in the LV Feeder IDs do not have the from_node the same as the to_node of the transformer"
    )


# test_check_line_transformer_nodes()


def test_check_timestamps():
    active_timestamp = pd.date_range(start="2025-06-09 08:00", end="2025-06-09 12:00", freq="h")
    reactive_timestamp = pd.date_range(start="2025-06-09 08:00", end="2025-06-09 12:00", freq="h")
    ev_timestamp = pd.date_range(start="2025-06-09 08:00", end="2025-06-09 13:00", freq="h")
    with pytest.raises(a3.NonMatchingTimestamps) as excinfo:
        a3.check_timestamps(active_timestamp, reactive_timestamp, ev_timestamp)
    assert str(excinfo.value) == "Timestamps between the active, reactive and ev profiles do not match"


# test_check_timestamps()


def test_check_profile_ids():
    active_ids = pd.Index(["1", "2", "4"])
    reactive_ids = pd.Index(["1", "2", "3"])
    with pytest.raises(a3.InvalidProfileIds) as excinfo:
        a3.check_profile_ids(active_ids, reactive_ids)
    assert str(excinfo.value) == "The active and reactive load profile IDs are not matching"


# test_check_profile_ids()


def test_check_symload_ids():
    active_ids = pd.Index(["1", "2"])
    reactive_ids = pd.Index(["3", "4"])
    symload_ids = pd.Index(["1", "2", "3", "5"])
    with pytest.raises(a3.InvalidSymloadIds) as excinfo:
        a3.check_symload_ids(active_ids, reactive_ids, symload_ids)
    assert str(excinfo.value) == "The active and reactive load profile IDs are not matching the sym_load IDs"


# test_check_symload_ids()


def test_check_number_of_ev_profiles():
    ev_profiles = pd.Index(["1", "2", "3"])
    symload_profiles = [10, 12]
    with pytest.raises(a3.InvalidNumberOfEVProfiles) as excinfo:
        a3.check_number_of_ev_profiles(ev_profiles, symload_profiles)
    assert str(excinfo.value) == "The number of EV charging profile is larger than the number of sym_loads"


# test_check_number_of_ev_profiles()


def test_check_valid_line_ids():
    id = 5
    Line_ids = [1, 2, 3, 4]
    with pytest.raises(a3.InvalidLineIds) as excinfo:
        a3.check_valid_line_ids(id, Line_ids)
    assert str(excinfo.value) == "Invalid Line ID"


# test_check_valid_line_ids()


def test_check_line_id_connected():
    fr = pd.Series(1)
    to = pd.Series(0)
    with pytest.raises(a3.NonConnnected) as excinfo:
        a3.check_line_id_connected(fr, to)
    assert str(excinfo.value) == "Line ID not connected at both sides in the base case"


# test_check_line_id_connected()


def test_find_alternative_lines():
    vertex_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    edge_ids = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    edge_vertex_id_pairs = [(1, 2), (2, 3), (2, 4), (4, 5), (1, 6), (6, 7), (6, 8), (8, 9), (4, 8), (0, 1)]
    edge_enabled = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    source_vertex_id = 0
    test = a1.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    assert test.find_alternative_edges(22) == [24]

    with pytest.raises(a1.IDNotFoundError) as excinfo:
        test.find_alternative_edges(26)
    assert str(excinfo.value) == "Disabled edge id not found in edge array"

    with pytest.raises(a1.EdgeAlreadyDisabledError) as excinfo:
        test.find_alternative_edges(24)
    assert str(excinfo.value) == "Edge is already disabled"


# test_find_alternative_lines()

# def test_power_flow_calc():
#     active_power_profile=pd.read_parquet("data/assignment 3 input/active_power_profile.parquet")
#     alt_lines_list=[24]
#     line_id_list=[16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
#     output_data=pd.DataFrame()
#     output_data['Alternative line ID']=24
#     output_data['Max_loading']=0.001631
#     output_data['Max_line_id']=17.0
#     output_data['Max_timestamp']='2025-01-01 08:15:00'
#     assert a3.power_flow_calc(active_power_profile,alt_lines_list,line_id_list)==output_data

# test_power_flow_calc()
