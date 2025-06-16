import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from power_grid_model.utils import json_deserialize, json_serialize

import power_system_simulation.assignment_1 as a1

# import power_system_simulation.assignment_2 as a2
import power_system_simulation.assignment_3 as a3
from power_system_simulation.assignment_3 import ComponentType, ev_penetration

with open("data/assignment_3_input/input_network_data.json") as fp:
    data = fp.read()

input_data = json_deserialize(data)

with open("data/assignment_3_input/meta_data.json") as fp:
    meta = fp.read()

meta_data = json.loads(meta)


active_power_profile = pd.read_parquet("data/assignment_3_input/active_power_profile.parquet")
reactive_power_profile = pd.read_parquet("data/assignment_3_input/reactive_power_profile.parquet")
ev_active_power_profile = pd.read_parquet("data/assignment_3_input/ev_active_power_profile.parquet")

dtype = {
    "names": [
        "id",
        "from_node",
        "to_node",
        "from_status",
        "to_status",
        "r1",
        "x1",
        "c1",
        "tan1",
        "r0",
        "x0",
        "c0",
        "tan0",
        "i_n",
    ]
}
df = pd.DataFrame(
    input_data[ComponentType.line], columns=dtype["names"]
)  # get the data for the lines of the grid as a dataframe


def test_check_source_transformer():
    meta = {"lv_busbar": 1, "lv_feeders": [16, 20], "mv_source_node": 0, "source": 10, "transformer": [11, 10]}
    with pytest.raises(a3.MoreThanOneTransformerOrSource):
        a3.check_source_transformer(meta)


# test_check_source_transformer()


def test_check_valid_LV_ids():
    LV_ids = [15, 20]
    Line_ids = [16, 17, 18, 19, 20, 21, 22, 23, 24]
    with pytest.raises(a3.InvalidLVIds):
        a3.check_valid_LV_ids(LV_ids, Line_ids)


# test_check_valid_LV_ids()


def test_check_line_transformer_nodes():
    lines_from_nodes = [1, 1]
    transformer_to_node = 2
    with pytest.raises(a3.NonMatchingTransformerLineNodes):
        a3.check_line_transformer_nodes(lines_from_nodes, transformer_to_node)


# test_check_line_transformer_nodes()


def test_check_timestamps():
    active_timestamp = pd.date_range(start="2025-06-09 08:00", end="2025-06-09 12:00", freq="h")
    reactive_timestamp = pd.date_range(start="2025-06-09 08:00", end="2025-06-09 12:00", freq="h")
    ev_timestamp = pd.date_range(start="2025-06-09 08:00", end="2025-06-09 13:00", freq="h")
    with pytest.raises(a3.NonMatchingTimestamps):
        a3.check_timestamps(active_timestamp, reactive_timestamp, ev_timestamp)


# test_check_timestamps()


def test_check_profile_ids():
    active_ids = pd.Index(["1", "2", "4"])
    reactive_ids = pd.Index(["1", "2", "3"])
    with pytest.raises(a3.InvalidProfileIds):
        a3.check_profile_ids(active_ids, reactive_ids)


# test_check_profile_ids()


def test_check_symload_ids():
    active_ids = pd.Index(["1", "2"])
    reactive_ids = pd.Index(["3", "4"])
    symload_ids = pd.Index(["1", "2", "3", "5"])
    with pytest.raises(a3.InvalidSymloadIds):
        a3.check_symload_ids(active_ids, reactive_ids, symload_ids)


# test_check_symload_ids()


def test_check_number_of_ev_profiles():
    ev_profiles = pd.Index(["1", "2", "3"])
    symload_profiles = [10, 12]
    with pytest.raises(a3.InvalidNumberOfEVProfiles):
        a3.check_number_of_ev_profiles(ev_profiles, symload_profiles)


# test_check_number_of_ev_profiles()


def test_check_valid_line_ids():
    id = 5
    Line_ids = [1, 2, 3, 4]
    with pytest.raises(a3.InvalidLineIds):
        a3.check_valid_line_ids(id, Line_ids)


# test_check_valid_line_ids()


def test_check_line_id_connected():
    fr = pd.Series(1)
    to = pd.Series(0)
    with pytest.raises(a3.NonConnnected):
        a3.check_line_id_connected(fr, to)


# test_check_line_id_connected()


def test_find_alternative_lines():
    vertex_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    edge_ids = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    edge_vertex_id_pairs = [(1, 2), (2, 3), (2, 4), (4, 5), (1, 6), (6, 7), (6, 8), (8, 9), (4, 8), (0, 1)]
    edge_enabled = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    source_vertex_id = 0
    test = a1.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    assert test.find_alternative_edges(22) == [24]

    with pytest.raises(a1.IDNotFoundError):
        test.find_alternative_edges(26)

    with pytest.raises(a1.EdgeAlreadyDisabledError):
        test.find_alternative_edges(24)


# test_find_alternative_lines()

# def test_power_flow_calc():
#     active_power_profile=pd.read_parquet("data/assignment_3_input/active_power_profile.parquet")
#     alt_lines_list=[24]
#     line_id_list=[16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
#     output_data=pd.DataFrame()
#     output_data['Alternative line ID']=24
#     output_data['Max_loading']=0.001631
#     output_data['Max_line_id']=17.0
#     output_data['Max_timestamp']='2025-01-01 08:15:00'
#     assert a3.power_flow_calc(active_power_profile,alt_lines_list,line_id_list)==output_data

# test_power_flow_calc()


def test_optimal_tap_position_criteria():
    criteria = "voltage"
    with pytest.raises(a3.InvalidCriteria):
        a3.optimal_tap_position(criteria)


def test_ev_assignment_basic():
    penetration_level = 1  # or fraction, depending on your logic
    output_node, output_line = ev_penetration(
        input_data, meta_data, ev_active_power_profile, penetration_level, random_seed=42
    )
    # assigned_df should have a row for every node
    assert len(output_node) == len(input_data["node"]) - 1


def test_no_ev_assigned_if_no_feeders():
    modified_meta = meta_data.copy()
    modified_meta["lv_feeders"] = []  # No feeders

    assigned_nodes, assigned_df = ev_penetration(
        input_data, modified_meta, ev_active_power_profile, penetration_level=5, random_seed=42
    )
    assert len(assigned_nodes) == 0
    assert assigned_df.empty
