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
import power_system_simulation.assignment_2 as a2


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


class InvalidLineIds(Exception):
    "Raised when the given Line ID to disconnect is not a valid"


class NonConnnected(Exception):
    "Raised when the given Line ID is not connected at both sides in the base case"


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


def check_valid_line_ids(id, Line_ids):
    if id not in Line_ids:
        raise InvalidLVIds("Invalid Line ID")


def check_line_id_connected(fr, to):
    if not (fr == [1] and to == [1]):
        raise NonConnnected("Line ID not connected at both sides in the base case")


def find_alternative_lines(
    vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id, id_to_disconnect
):
    test = a1.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    return test.find_alternative_edges(id_to_disconnect)


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


def N_minus_one_calculation(id_to_disconnect):
    check_valid_line_ids(id_to_disconnect, input_data[ComponentType.line]["id"])  # check if ID is valid
    check_line_id_connected(
        df[df["id"] == id_to_disconnect]["from_status"].tolist(), df[df["id"] == id_to_disconnect]["to_status"].tolist()
    )  # check if line with selected ID is connected
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
    status_list = list(df["to_status"].tolist()) + list(
        input_data[ComponentType.transformer]["to_status"].tolist()
    )  # add transformer connection status to list of lines' statuses
    line_id_list = df["id"].tolist()
    new_id = (df["id"].iloc[-1] + 1).tolist()
    line_id_list.append(new_id)  # add another line id to mimic the transformer connection
    # print(input_data[ComponentType.node]['id'].tolist())
    # print(line_id_list)
    # print(line_nodes_id_pairs)
    # print(status_list)
    # print(meta_data['mv_source_node'])
    print(
        f"To make the grid fully connected, the following lines need to be connected: {find_alternative_lines(input_data[ComponentType.node]['id'].tolist(), line_id_list, line_nodes_id_pairs, status_list, meta_data['mv_source_node'],id_to_disconnect)}"
    )  # find alternative currently disconnected lines to make the grid fully connected

def optimal_tap_position(criteria = "Loss"):
    if criteria not in ["Voltage_deviation", "Total_loss"]:
        raise ValueError("Invalid criteria specified. Use 'Voltage_deviation' or 'Total_loss'.")
    
    transformer_data = input_data[ComponentType.transformer]
    tap_min, tap_max = int(transformer_data["tap_min"][0]), int(transformer_data["tap_max"][0])
    tap_steps = list(range(tap_max, tap_min + 1))
    # display("tap_min:", tap_min)
    # display("tap_max:", tap_max)
    # display("tap_steps:", tap_steps)
    id =  transformer_data["id"]
    total_loss = []
    total_deviation = []
    for tap in tap_steps:
        update_data = a2.prepare_update_data(active_power_profile, reactive_power_profile, tap, id, 'both')
        output_data = a2.calculate_power_flow(input_data, update_data)

        output_line = a2.calculate_line_stats(output_data, active_power_profile)
        output_node = a2.calculate_node_stats(output_data, active_power_profile)
        # a2.display_results(output_node, output_line)

        total_loss.append(output_line["Total_loss"].sum())

        deviation_max = abs(output_node["Max_voltage"]-1).sum()
        deviation_min = abs(output_node["Min_voltage"]-1).sum()
        total_deviation.append((deviation_max + deviation_min)/(len(output_node)*2)) # Divide total deviation by number of nodes (max and min) and number of timestamps

    if criteria == "Voltage_deviation":
        display("Optimal tap position based on voltage deviation:")
        optimal_tap = tap_steps[np.argmin(total_deviation)]
        display(f"Optimal tap position: {optimal_tap}, with total deviation: {min(total_deviation)}")
        # display("Total deviation for each tap position:")
        # display(pd.DataFrame({"tap_position": tap_steps, "total_deviation": total_deviation}))

    elif criteria == "Total_loss":
        display("Optimal tap position based on total loss:")
        optimal_tap = tap_steps[np.argmin(total_loss)]
        display(f"Optimal tap position: {optimal_tap}, with total loss: {min(total_loss)}")
        # display("Total losses for each tap position:")
        # display(pd.DataFrame({"tap_position": tap_steps, "total_loss": total_loss}))


with open("data/assignment 3 input/input_network_data.json") as fp:
    data = fp.read()

input_data = json_deserialize(data)

with open("data/assignment 3 input/meta_data.json") as fp:
    meta = fp.read()

meta_data = json.loads(meta)
# pprint.pprint(meta_data)

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
# display("Input data:")
# display(input_data[ComponentType.line])
# display("\n\n")
# input_data_validity_check(input_data, meta_data)

optimal_tap_position("Voltage_deviation")
id_to_disconnect = 22
# N_minus_one_calculation(id_to_disconnect)
pass