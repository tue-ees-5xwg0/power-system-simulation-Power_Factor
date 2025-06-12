import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from power_grid_model import PowerGridModel  # may be monkeypatched

from power_system_simulation.assignment_2 import (
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
    # Create a DataFrame with 2 timestamps and 3 columns
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
    # Suppose 2 timestamps, 3 nodes
    u_pu = np.array([[1.0, 0.9, 1.1], [1.05, 0.95, 1.0]])
    # For lines: 2 timestamps, 2 lines
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


# 2. Tests for calculate_node_stats
def test_calculate_node_stats(dummy_output_data, dummy_batch_profile):
    df = calculate_node_stats(dummy_output_data, dummy_batch_profile)
    # For timestamp 0: row [1.0,0.9,1.1] → min=0.9 at idx=1 → node id = 1+1=2; max=1.1 at idx=2 → 3
    # For timestamp 1: row [1.05,0.95,1.0] → min=0.95 at idx=1 → 2; max=1.05 at idx=0 → 1
    expected = pd.DataFrame(
        {
            "id": dummy_batch_profile.index,
            "Max_voltage": [1.1, 1.05],
            "Max_voltage_node": [3, 1],
            "Min_voltage": [0.9, 0.95],
            "Min_voltage_node": [2, 2],
        }
    )
    # Ensure same dtypes; timestamps equal
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected.reset_index(drop=True))


# 3. Tests for calculate_line_stats
def test_calculate_line_stats(dummy_output_data, dummy_batch_profile):
    df = calculate_line_stats(dummy_output_data, dummy_batch_profile)
    # Compute expected manually:
    # For line 101 (col 0): loading [0.5,0.6] → min=0.5 at ts idx 0, max=0.6 at idx1
    # timestamp values:
    ts = dummy_batch_profile.index
    exp_min_ts = ts[0]
    exp_max_ts = ts[1]
    # total_loss for line 101: trapezoid of p_from+p_to: sums = [19, 23], dx=3600e9, integral = (19+23)/2 * dx = 21 * 3600e9; then / (3.6e15) = (21*3600e9)/(3.6e15) = 21*(3600/3.6)*(1e9/1e15) =21*1000*(1e-6)=21*0.001=0.021? Let's compute precisely in test.
    # Similarly for line 102.
    # We can compute in test to avoid mistakes:
    import numpy as _np

    sums = dummy_output_data[ComponentType.line]["p_from"] + dummy_output_data[ComponentType.line]["p_to"]
    total_loss = _np.trapezoid(sums, dx=3600 * (10**9), axis=0) / (3.6 * (10**15))
    expected = pd.DataFrame(
        {
            "id": np.array([101, 102]),
            "Total_loss": total_loss,
            "Max_loading": [0.6, 0.7],  # careful: col ordering: loading[:,0]=[0.5,0.6], loading[:,1]=[0.7,0.65]
            "Max_loading_timestamp": [ts[1], ts[0]],  # for col1: max=0.7 at idx0
            "Min_loading": [0.5, 0.65],
            "Min_loading_timestamp": [ts[0], ts[1]],  # for col1: min=0.65 at idx1
        }
    )
    # Reorder expected rows if needed to match ordering in function (ids from np.unique → sorted [101,102])
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected.reset_index(drop=True))


# # 4. Tests for prepare_update_data
# def test_prepare_update_data(dummy_batch_profile):
#     upd = prepare_update_data(dummy_batch_profile, dummy_batch_profile, "active")
#     # Expect dict with key ComponentType.sym_load
#     arr = upd.get(ComponentType.sym_load)
#     assert arr is not None
#     # Check fields
#     # 'id' equals columns as numpy array
#     np.testing.assert_array_equal(arr["id"], dummy_batch_profile.columns.to_numpy())
#     np.testing.assert_array_equal(arr["p_specified"], dummy_batch_profile.to_numpy())
#     # q_specified zeros
#     assert np.all(arr["q_specified"] == 0.0)

# # 6. Tests for load_batch_profiles
# def test_load_batch_profiles_success(tmp_path):
#     # Create two small DataFrames with same index
#     idx = pd.date_range("2025-01-01", periods=2, freq="H")
#     df1 = pd.DataFrame({"X": [1,2]}, index=idx)
#     df2 = pd.DataFrame({"Y": [3,4]}, index=idx)
#     p1 = tmp_path / "a.parquet"
#     p2 = tmp_path / "b.parquet"
#     df1.to_parquet(p1)
#     df2.to_parquet(p2)
#     a, b = load_batch_profiles(str(p1), str(p2))
#     pd.testing.assert_frame_equal(a, df1)
#     pd.testing.assert_frame_equal(b, df2)


def test_load_batch_profiles_mismatch(tmp_path):
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


def test_calculate_power_flow_success(monkeypatch):
    # Prepare dummy input_data and update_data
    dummy_input = {"whatever": "value"}
    dummy_update = {"x": "y"}
    # Monkeypatch assert_valid_batch_data to no-op
    import power_system_simulation.assignment_2

    monkeypatch.setattr(power_system_simulation.assignment_2, "assert_valid_batch_data", lambda **kwargs: None)

    # Monkeypatch PowerGridModel to return dummy output_data
    class DummyModel:
        def __init__(self, input_data):
            assert input_data is dummy_input

        def calculate_power_flow(self, update_data, calculation_method):
            assert update_data is dummy_update
            return {"dummy": "out"}

    monkeypatch.setattr(power_system_simulation.assignment_2, "PowerGridModel", DummyModel)
    result = calculate_power_flow(input_data=dummy_input, update_data=dummy_update)
    assert result == {"dummy": "out"}


def test_calculate_power_flow_validation_error(monkeypatch):
    dummy_input = {}
    dummy_update = {}
    import power_system_simulation.assignment_2

    # Make assert_valid_batch_data raise ValidationException with .errors
    class FakeError:
        def __init__(self):
            self.component = "comp"
            self.ids = [1, 2]

    def fake_assert(**kwargs):
        ve = ValidationException("bad")
        ve.errors = [FakeError()]
        raise ve

    monkeypatch.setattr(power_system_simulation.assignment_2, "assert_valid_batch_data", fake_assert)
    # Ensure PowerGridModel is not called
    with pytest.raises(ValidationException):
        calculate_power_flow(input_data=dummy_input, update_data=dummy_update)
