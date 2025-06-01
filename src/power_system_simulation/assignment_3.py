# some basic imports
import json
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from power_grid_model import (
    BranchSide,
    CalculationMethod,
    CalculationType,
    ComponentType,
    DatasetType,
    LoadGenType,
    MeasuredTerminalType,
    PowerGridModel,
    TapChangingStrategy,
    initialize_array,
)
from power_grid_model.utils import json_deserialize, json_serialize
from power_grid_model.validation import ValidationException, assert_valid_batch_data, assert_valid_input_data

import power_system_simulation.assignment_1 as a1


class MoreThanOneTransformerOrSource(Exception):
    "Raised when there is more than one transformer or source in meta_data.json"


class InvalidLVIds(Exception):
    "Raised when LV Feeder IDs are not valid line IDs."


class NonMatchingTransformerLineNodes(Exception):
    "Raised when the lines in the LV Feeder IDs do not have the from_node the same as the to_node of the transformer."


class NonMatchingTimestamps(Exception):
    "Raised when the timestamps are not matching between the active load profile, reactive load profile, and EV charging profile."


class InvalidProfileIds(Exception):
    "Raised when the IDs in active load profile and reactive load profile are not matching."


class InvalidSymloadIds(Exception):
    "Raised when the IDs in active load profile and reactive load profile are not matching the symload IDs."


class InvalidNumberOfEVProfiles(Exception):
    "raised when the number of EV charging profile is at least the same as the number of sym_load."


def check_source_transformer(meta_data):
    if (len([meta_data["source"]]) != 1) or (len([meta_data["transformer"]]) != 1):
        raise MoreThanOneTransformerOrSource("LV grid contains more than one source or transformer")


def check_valid_LV_ids(LV_ids, Line_ids):
    if not all(item in Line_ids for item in LV_ids):
        raise InvalidLVIds("LV feeders contain invalid ids")


def check_line_transformer_nodes(lines_from_nodes, transformer_to_node):
    if not all(element == transformer_to_node for element in lines_from_nodes):
        raise NonMatchingTransformerLineNodes(
            "The lines in the LV Feeder IDs do not have the from_node the same as the to_node of the transformer"
        )


def check_timestamps(active_timestamp, reactive_timestamp, ev_timestamp):
    if not (active_timestamp.equals(reactive_timestamp) and active_timestamp.equals(ev_timestamp)):
        raise NonMatchingTimestamps("Timestamps between the active, reactive and ev profiles do not match")


def check_profile_ids(active_ids, reactive_ids):
    if not active_ids.equals(reactive_ids):
        raise InvalidProfileIds("The active and reactive load profile IDs are not matching")


def check_symload_ids(active_ids, reactive_ids, symload_ids):
    if not (all(item in symload_ids for item in active_ids) and (all(item in symload_ids for item in reactive_ids))):
        raise InvalidSymloadIds("The active and reactive load profile IDs are not matching the sym_load IDs")


def check_number_of_ev_profiles(ev_profiles, symload_profiles):
    if len(ev_profiles) > len(symload_profiles):
        raise InvalidNumberOfEVProfiles("The number of EV charging profile is larger than the number of sym_loads")


def input_data_validity_check(input_data, meta_data):

    assert_valid_input_data(
        input_data=input_data, calculation_type=CalculationType.power_flow
    )  # check if input data is valid
    check_source_transformer(meta_data)  # check if LV grid has exactly one transformer, and one source.
    # print(type(input_data)) #dict
    print(input_data)  # ComponentType.line
    check_valid_LV_ids(
        meta_data["lv_feeders"], input_data[ComponentType.line]["id"]
    )  # check if All IDs in the LV Feeder IDs are valid line IDs

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
    check_line_transformer_nodes(
        df[df["id"].isin(meta_data["lv_feeders"])]["from_node"].tolist(),
        input_data[ComponentType.transformer]["to_node"],
    )  # check if all the lines in the LV Feeder IDs have the from_node the same as the to_node of the transformer

    transformer_tuple = list(
        zip(
            input_data[ComponentType.transformer]["from_node"].tolist(),
            input_data[ComponentType.transformer]["to_node"].tolist(),
        )
    )  # transformer also connects two nodes
    line_nodes_id_pairs = (
        list(
            zip(
                input_data[ComponentType.line]["from_node"].tolist(), input_data[ComponentType.line]["to_node"].tolist()
            )
        )
        + transformer_tuple
    )  # add nodes connected by transformer to list of lines connecting nodes
    status_list = list(input_data[ComponentType.line]["to_status"].tolist()) + list(
        input_data[ComponentType.transformer]["to_status"].tolist()
    )  # add transformer connection status to list of lines' statuses
    a1.check_connect(
        input_data[ComponentType.node]["id"], status_list, line_nodes_id_pairs
    )  # check if the grid is fully connected in the initial state
    a1.check_cycle(status_list, line_nodes_id_pairs)  # check if the grid has no cycles in the initial state

    active_power_profile = pd.read_parquet("data/assignment 3 input/active_power_profile.parquet")
    reactive_power_profile = pd.read_parquet("data/assignment 3 input/reactive_power_profile.parquet")
    ev_active_power_profile = pd.read_parquet("data/assignment 3 input/ev_active_power_profile.parquet")

    # print(active_power_profile)
    # print(reactive_power_profile)
    # print(ev_active_power_profile)
    # print(input_data[ComponentType.sym_load]['id'],active_power_profile.columns,reactive_power_profile.columns)
    check_timestamps(
        active_power_profile.index, reactive_power_profile.index, ev_active_power_profile.index
    )  # checks if timestamps are matching
    check_profile_ids(
        active_power_profile.columns, reactive_power_profile.columns
    )  # checks if number of active and reactive profiles are matching
    check_symload_ids(
        active_power_profile.columns, reactive_power_profile.columns, input_data[ComponentType.sym_load]["id"]
    )  # checks if IDs are matching
    check_number_of_ev_profiles(
        ev_active_power_profile.columns, input_data[ComponentType.sym_load]["id"]
    )  # checks if number of EV profiles does not exceed number of sym_loads

    # DOUBLE CHECK check_number_of_ev_profiles FUNCTION!!!


with open("data/assignment 3 input/input_network_data.json") as fp:
    data = fp.read()

input_data = json_deserialize(data)

with open("data/assignment 3 input/meta_data.json") as fp:
    meta = fp.read()

meta_data = json.loads(meta)
pprint.pprint(meta_data)

input_data_validity_check(input_data, meta_data)

pass
