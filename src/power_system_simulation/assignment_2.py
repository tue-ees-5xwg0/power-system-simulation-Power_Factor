import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from pandas import DataFrame
from power_grid_model import (
    CalculationMethod,
    CalculationType,
    ComponentType,
    DatasetType,
    PowerGridModel,
    initialize_array,
)
from power_grid_model.utils import json_deserialize, json_serialize
from power_grid_model.validation import ValidationException, assert_valid_batch_data, assert_valid_input_data


def load_input_data(filepath: str):
    """Read and deserialize input network data from a JSON file."""
    try:
        with open(filepath) as fp:
            try:
                data = fp.read()
            except Exception:
                raise ValidationException("Error reading input data file")
    except Exception:
        raise ValidationException("Error reading input data file")
    try:
        input_data = json_deserialize(data)
    except Exception:
        raise ValidationException("Error deserializing input data")
    return input_data


def load_batch_profiles(active_profile_path: str, reactive_profile_path: str):
    """
    Read active and reactive power profiles from parquet files,
    check matching timestamps, and return them.
    """
    try:
        active_power_profile = pd.read_parquet(active_profile_path)
        reactive_power_profile = pd.read_parquet(reactive_profile_path)
    except Exception as e:
        raise ValidationException(f"Error reading batch profile files: {e}")
    if not active_power_profile.index.equals(reactive_power_profile.index):
        raise ValidationException("Active and reactive batch data have different timestamps")
    return active_power_profile, reactive_power_profile


# def prepare_update_data(
#     active_batch_profile: pd.DataFrame, reactive_batch_profile: pd.DataFrame, profile_type: str = "active"
# ):
#     """
#     Given a DataFrame of power profile indexed by timestamps and columns=IDs,
#     build the update_data dict for sym_load.
#     """
#     # initialize the underlying array
#     load_profile = initialize_array(DatasetType.update, ComponentType.sym_load, active_batch_profile.shape)

#     # assign IDs (allowing non-int IDs)
#     ids = active_batch_profile.columns.to_numpy()
#     try:
#         load_profile["id"] = ids
#     except (ValueError, TypeError):
#         load_profile["id"] = np.array(ids, dtype=object)

#     # fill the appropriate field
#     data_active = active_batch_profile.to_numpy()
#     data_reactive = reactive_batch_profile.to_numpy()
#     if profile_type == "active":
#         load_profile["p_specified"] = data_active
#         load_profile["q_specified"] = 0.0
#     elif profile_type == "reactive":
#         load_profile["p_specified"] = 0.0
#         load_profile["q_specified"] = data_reactive
#     elif profile_type == "both":
#         load_profile["p_specified"] = data_active
#         load_profile["q_specified"] = data_reactive
#     else:
#         raise ValueError(f"Unknown profile_type '{profile_type}', must be 'active' or 'reactive'")

#     return {ComponentType.sym_load: load_profile}


def prepare_update_data(
    active_batch_profile: pd.DataFrame,
    reactive_batch_profile: pd.DataFrame,
    tap_pos: int = -1,
    transformer_id: int = -1,
    profile_type: str = "active",
):
    """
    Given:
      - active_batch_profile: DataFrame indexed by timestamps, columns=house IDs → active p
      - reactive_batch_profile: same shape → reactive q
      - transformer_id: the int ID of the MV/LV transformer
      - tap_pos: the integer tap position to use for every time step
    Build and return the full update_data dict for:
      1) sym_load (p_specified, q_specified)
      2) transformer (id, tap_pos, from_status, to_status)
    """
    # 1) build sym_load update
    load_profile = initialize_array(DatasetType.update, ComponentType.sym_load, active_batch_profile.shape)

    # assign the sym_load IDs
    ids = active_batch_profile.columns.to_numpy()
    try:
        load_profile["id"] = ids
    except (ValueError, TypeError):
        load_profile["id"] = np.array(ids, dtype=object)

    # fill p_specified / q_specified
    data_active = active_batch_profile.to_numpy()
    data_reactive = reactive_batch_profile.to_numpy()
    if profile_type == "active":
        load_profile["p_specified"] = data_active
        load_profile["q_specified"] = 0.0
    elif profile_type == "reactive":
        load_profile["p_specified"] = 0.0
        load_profile["q_specified"] = data_reactive
    elif profile_type == "both":
        load_profile["p_specified"] = data_active
        load_profile["q_specified"] = data_reactive
    else:
        raise ValueError(f"Unknown profile_type '{profile_type}', must be 'active', 'reactive', or 'both'")

    # start composing the return dict
    update_data = {ComponentType.sym_load: load_profile}

    # 2) optionally build transformer update
    if transformer_id >= 0:
        # initialize the transformer update array
        transformer_update = initialize_array(
            DatasetType.update, ComponentType.transformer, (active_batch_profile.shape[0], 1)
        )

        # assign the transformer ID
        transformer_update["id"] = np.array([transformer_id], dtype=int)

        # set the tap position for all timestamps
        transformer_update["tap_pos"] = tap_pos

        # add to update_data
        update_data[ComponentType.transformer] = transformer_update

    return update_data


def calculate_power_flow(input_data, update_data):
    """
    Assert validity and run the power flow calculation, returning output_data.
    """
    try:
        assert_valid_batch_data(
            input_data=input_data, update_data=update_data, calculation_type=CalculationType.power_flow
        )
        model = PowerGridModel(input_data=input_data)
        output_data = model.calculate_power_flow(
            update_data=update_data, calculation_method=CalculationMethod.newton_raphson
        )
    except ValidationException as ex:
        # Print each error found
        for error in ex.errors:
            print(type(error).__name__, error.component, ":", error.ids)
        # Re-raise or return None so caller knows computation did not succeed
        raise
    return output_data


def calculate_node_stats(output_data, batch_profile: pd.DataFrame):
    """
    For each node in batch_profile.index, compute min/max voltages and corresponding node indices.
    Returns a DataFrame with columns: ["id", "Max_voltage", "Max_voltage_node", "Min_voltage", "Min_voltage_node"].
    """
    ids_node = batch_profile.index
    n = len(ids_node)
    min_voltage = np.zeros(n)
    max_voltage = np.zeros(n)
    min_voltage_id = np.zeros(n, dtype=int)
    max_voltage_id = np.zeros(n, dtype=int)

    # output_data[ComponentType.node]["u_pu"] assumed shape (timestamps, nodes)
    u_pu = output_data[ComponentType.node]["u_pu"]
    # For each timestamp index i, find min/max along the node axis
    for i in range(n):
        row = u_pu[i, :]
        min_idx = row.argmin()
        max_idx = row.argmax()
        min_voltage[i] = row[min_idx]
        max_voltage[i] = row[max_idx]
        # +1 to handle 0-based indexing in Python
        min_voltage_id[i] = min_idx + 1
        max_voltage_id[i] = max_idx + 1

    output_node = pd.DataFrame(
        {
            "id": ids_node,
            "Max_voltage": max_voltage,
            "Max_voltage_node": max_voltage_id,
            "Min_voltage": min_voltage,
            "Min_voltage_node": min_voltage_id,
        }
    )
    return output_node


def calculate_line_stats(output_data, batch_profile: pd.DataFrame):
    """
    For each line ID, compute min/max loading with timestamps, and total energy loss.
    Returns a DataFrame with columns:
      ["id", "Total_loss", "Max_loading", "Max_loading_timestamp", "Min_loading", "Min_loading_timestamp"].
    """
    # Unique line IDs
    ids = np.unique(output_data[ComponentType.line]["id"])
    n_lines = len(ids)
    # Prepare arrays
    min_loading = np.zeros(n_lines)
    max_loading = np.zeros(n_lines)
    min_loading_timestamp = np.empty(n_lines, dtype=object)
    max_loading_timestamp = np.empty(n_lines, dtype=object)

    loading = output_data[ComponentType.line]["loading"]  # shape (timestamps, lines)
    timestamps = batch_profile.index

    # Compute min/max and find first matching timestamp
    for i in range(n_lines):
        col = loading[:, i]
        min_val = col.min()
        max_val = col.max()
        min_loading[i] = min_val
        max_loading[i] = max_val
        # find first timestamp where equals
        # Using .argwhere or simple loop; for clarity:
        # Note: if multiple matches, picks the first occurrence
        min_idx = int(np.where(col == min_val)[0][0])
        max_idx = int(np.where(col == max_val)[0][0])
        min_loading_timestamp[i] = timestamps[min_idx]
        max_loading_timestamp[i] = timestamps[max_idx]

    # Compute total energy loss with trapezoidal rule:
    p_from = output_data[ComponentType.line]["p_from"]
    p_to = output_data[ComponentType.line]["p_to"]
    total_loss = np.trapezoid(p_from + p_to, dx=3600 * (10**9), axis=0) / (3.6 * (10**15))

    output_line = pd.DataFrame(
        {
            "id": ids,
            "Total_loss": total_loss,
            "Max_loading": max_loading,
            "Max_loading_timestamp": max_loading_timestamp,
            "Min_loading": min_loading,
            "Min_loading_timestamp": min_loading_timestamp,
        }
    )
    return output_line


def display_results(output_node: pd.DataFrame, output_line: pd.DataFrame):
    """Display the node and line result tables."""
    print("\nSelf-made output node table: ")
    display(output_node)
    print("\nSelf-made line table: ")
    display(output_line)


def compare_with_expected(timestamp_table_path: str, line_table_path: str):
    """Read expected output tables and display them for comparison."""
    try:
        expected_output_timestamp = pd.read_parquet(timestamp_table_path)
        expected_output_line = pd.read_parquet(line_table_path)
    except Exception as e:
        raise ValidationException(f"Error reading expected output files: {e}")

    # print("\nProvided timestamp table: ")
    # display(expected_output_timestamp)
    # print("\nProvided line table: ")
    # display(expected_output_line)


def power_flow_results(update_data, batch_profile):
    """
    This function performs an analysis of the power flow results by:
      - Asserting and computing power flow
      - Calculating node stats
      - Calculating line stats (including energy loss)
      - Displaying results and comparing with expected tables
    """
    # Compute power flow
    try:
        output_data = calculate_power_flow(input_data=input_data, update_data=update_data)
    except ValidationException:
        return

    # Node calculation
    output_node = calculate_node_stats(output_data, batch_profile)

    # Line calculation
    output_line = calculate_line_stats(output_data, batch_profile)

    display_results(output_node, output_line)

    # Compare with expected outputs
    compare_with_expected(
        "data/expected_output/output_table_row_per_timestamp.parquet",
        "data/expected_output/output_table_row_per_line.parquet",
    )


def main():
    """Executes all the functions: load data, prepare update_data, and call power_flow_results."""
    # Load input network data
    input_filepath = "data/input/input_network_data.json"
    try:
        global input_data  # so power_flow_results can access it by name as in original code
        input_data = load_input_data(input_filepath)
    except ValidationException as e:
        print(f"Failed to load input data: {e}")
        return

    try:
        line_df = DataFrame(input_data[ComponentType.line])
        # display(line_df)
    except Exception:
        pass

    # Load batch profiles
    try:
        active_power_profile, reactive_power_profile = load_batch_profiles(
            "data/input/active_power_profile.parquet",
            "data/input/reactive_power_profile.parquet",
        )
    except ValidationException as e:
        print(f"Failed to load batch profiles: {e}")
        return

    # Display profiles
    # display(active_power_profile)
    # display(reactive_power_profile)
    # Prepare update_data for active profile, you can also add reactive now
    update_data = prepare_update_data(active_power_profile, reactive_power_profile, profile_type="both")
    # Run the analysis
    power_flow_results(update_data, active_power_profile)


if __name__ == "__main__":
    main()
