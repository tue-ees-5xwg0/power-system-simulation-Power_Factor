import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from power_grid_model import PowerGridModel  # may be monkeypatched

from power_system_simulation.assignment_2 import (
    CalculationMethod,
    ComponentType,
    ValidationException,
    calculate_line_stats,
    calculate_node_stats,
    calculate_power_flow,
    load_batch_profiles,
    load_input_data,
    power_flow_results,
    prepare_update_data,
)


# Fixtures which provide dummy data for testing
@pytest.fixture
def dummy_batch_profile(tmp_path):
    """Create a dummy batch profile DataFrame for testing."""
    idx = pd.DatetimeIndex(["2025-01-01T00:00", "2025-01-01T01:00"])
    df = pd.DataFrame(
        {
            1: [1.0, 2.0],
            2: [1.5, 2.5],
            3: [0.5, 1.5],
        },
        index=idx,
    )
    return df


@pytest.fixture
def dummy_output_data(dummy_batch_profile):
    """Create dummy data of the powerflow solution output for testing."""
    u_pu = np.array([[1.0, 0.9, 1.1], [1.05, 0.95, 1.0]])
    ids = np.array([101, 102])
    loading = np.array([[0.5, 0.7], [0.6, 0.65]])
    p_from = np.array([[10.0, 20.0], [12.0, 18.0]])
    p_to = np.array([[9.0, 19.0], [11.0, 17.0]])
    return {
        ComponentType.node: {"u_pu": u_pu},
        ComponentType.line: {
            "id": ids,
            "loading": loading,
            "p_from": p_from,
            "p_to": p_to,
        },
    }


def test_calculate_node_stats(dummy_output_data, dummy_batch_profile):
    """Test calculate_node_stats with dummy output data."""
    df = calculate_node_stats(dummy_output_data, dummy_batch_profile)

    expected = pd.DataFrame(
        {
            "id": dummy_batch_profile.index,
            "Max_voltage": [1.1, 1.05],
            "Max_voltage_node": [3, 1],
            "Min_voltage": [0.9, 0.95],
            "Min_voltage_node": [2, 2],
        }
    )
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected.reset_index(drop=True))


def test_calculate_line_stats(dummy_output_data, dummy_batch_profile):
    """Test calculate_line_stats with dummy output data."""
    df = calculate_line_stats(dummy_output_data, dummy_batch_profile)
    ts = dummy_batch_profile.index

    sums = dummy_output_data[ComponentType.line]["p_from"] + dummy_output_data[ComponentType.line]["p_to"]
    total_loss = np.trapezoid(sums, dx=3600 * (10**9), axis=0) / (3.6 * (10**15))
    expected = pd.DataFrame(
        {
            "id": np.array([101, 102]),
            "Total_loss": total_loss,
            "Max_loading": [0.6, 0.7],
            "Max_loading_timestamp": [ts[1], ts[0]],
            "Min_loading": [0.5, 0.65],
            "Min_loading_timestamp": [ts[0], ts[1]],
        }
    )
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected.reset_index(drop=True))


def test_prepare_update_data(dummy_batch_profile):
    """Test prepare_update_data with active profile type."""
    # supply both active & reactive, but only fill active
    upd = prepare_update_data(dummy_batch_profile, dummy_batch_profile, profile_type="active")
    arr = upd.get(ComponentType.sym_load)
    assert arr is not None

    # arr["id"] should be shape (n_timestamps, n_ids)
    ids = dummy_batch_profile.columns.to_numpy()
    n_t, n_ids = dummy_batch_profile.shape
    assert arr["id"].shape == (n_t, n_ids)

    # every row of arr["id"] must equal the column IDs
    for row in arr["id"]:
        np.testing.assert_array_equal(row, ids)

    # p_specified is also shape (n_timestamps, n_ids)
    np.testing.assert_array_equal(arr["p_specified"], dummy_batch_profile.to_numpy())

    # q_specified should be all zeros, same shape
    assert arr["q_specified"].shape == (n_t, n_ids)
    assert np.all(arr["q_specified"] == 0.0)


def test_load_batch_profiles_success(tmp_path):  # tmp_path is a pytest fixture that provides a temporary directory
    """Test loading two batch profiles from parquet files."""
    # Create two small DataFrames with same index
    idx = pd.date_range("2025-01-01", periods=2, freq="H")
    df1 = pd.DataFrame({"X": [1, 2]}, index=idx)
    df2 = pd.DataFrame({"Y": [3, 4]}, index=idx)
    # print(f"DataFrame 1:\n{df1}\nDataFrame 2:\n{df2}")
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    df1.to_parquet(p1)
    df2.to_parquet(p2)
    a, b = load_batch_profiles(str(p1), str(p2))
    pd.testing.assert_index_equal(a.index, b.index)


def test_load_batch_profiles_mismatch(tmp_path):
    """Test loading two batch profiles with different indices raises ValidationException."""
    idx1 = pd.date_range("2025-01-01", periods=2, freq="H")
    idx2 = pd.date_range("2025-01-02", periods=2, freq="H")
    df1 = pd.DataFrame({"X": [1, 2]}, index=idx1)
    df2 = pd.DataFrame({"Y": [3, 4]}, index=idx2)
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    df1.to_parquet(p1)
    df2.to_parquet(p2)
    with pytest.raises(ValidationException):
        load_batch_profiles(str(p1), str(p2))


# def test_calculate_power_flow_success(monkeypatch):
#     """Test calculate_power_flow with dummy input and update data."""
#     dummy_input = {"whatever": "value"}
#     dummy_update = {"x": "y"}

#     import power_system_simulation.assignment_2
#     monkeypatch.setattr(power_system_simulation.assignment_2, "assert_valid_batch_data", lambda **kwargs: None)

#     # Monkeypatch PowerGridModel to return dummy output_data
#     class DummyModel:
#         def __init__(self, input_data):
#             assert input_data is dummy_input

#         def calculate_power_flow(self, update_data, calculation_method):
#             assert update_data is dummy_update
#             return {"dummy": "out"}

#     monkeypatch.setattr(power_system_simulation.assignment_2, "PowerGridModel", DummyModel)
#     result = calculate_power_flow(input_data=dummy_input, update_data=dummy_update)
#     assert result == {"dummy": "out"}


# def test_calculate_power_flow_validation_error(monkeypatch):
#     """Test calculate_power_flow raises ValidationException when input data is invalid."""
#     dummy_input = {}
#     dummy_update = {}
#     import power_system_simulation.assignment_2

#     # Make assert_valid_batch_data raise ValidationException with .errors
#     class FakeError:
#         def __init__(self):
#             self.component = "comp"
#             self.ids = [1, 2]

#     def fake_assert(**kwargs):
#         ve = ValidationException("bad")
#         ve.errors = [FakeError()]
#         raise ve

#     monkeypatch.setattr(power_system_simulation.assignment_2, "assert_valid_batch_data", fake_assert)
#     # Ensure PowerGridModel is not called
#     with pytest.raises(ValidationException):
#         calculate_power_flow(input_data=dummy_input, update_data=dummy_update)
