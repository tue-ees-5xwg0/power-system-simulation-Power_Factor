from power_system_simulation.assignment_3 import ev_penetration, ComponentType
import pytest
import pandas as pd
import json
from power_grid_model.utils import json_deserialize, json_serialize

with open("data/assignment 3 input/input_network_data.json") as fp:
    data = fp.read()

input_data = json_deserialize(data)

with open("data/assignment 3 input/meta_data.json") as fp:
    meta = fp.read()

meta_data = json.loads(meta)


active_power_profile = pd.read_parquet("data/assignment 3 input/active_power_profile.parquet")
reactive_power_profile = pd.read_parquet("data/assignment 3 input/reactive_power_profile.parquet")
ev_active_power_profile = pd.read_parquet("data/assignment 3 input/ev_active_power_profile.parquet")

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



def test_ev_assignment_basic():
    penetration_level = 1  # or fraction, depending on your logic
    output_node, output_line = ev_penetration(
        input_data,
        meta_data,
        ev_active_power_profile,
        penetration_level,
        random_seed=42
    )
    # assigned_df should have a row for every node
    assert len(output_node) == len(input_data["node"]) - 1


def test_no_ev_assigned_if_no_feeders():
    modified_meta = meta_data.copy()
    modified_meta["lv_feeders"] = []  # No feeders
    
    assigned_nodes, assigned_df = ev_penetration(
        input_data,
        modified_meta,
        ev_active_power_profile,
        penetration_level=5,
        random_seed=42
    )
    assert len(assigned_nodes) == 0
    assert assigned_df.empty

