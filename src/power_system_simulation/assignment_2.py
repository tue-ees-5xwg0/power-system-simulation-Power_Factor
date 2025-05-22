import time
from typing import Dict
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from IPython.display import display
from power_grid_model.validation import ValidationException

import json
import pprint
from power_grid_model import (
    PowerGridModel,
    CalculationType,
    CalculationMethod,
    ComponentType,
    DatasetType,
    initialize_array
)
from power_grid_model.utils import json_deserialize, json_serialize
from power_grid_model.validation import (
    assert_valid_input_data,
    assert_valid_batch_data
)
class ValidationException(Exception):
    """Raised when input data is invalid"""
    
class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        print(f'Execution time for {self.name} is {(time.perf_counter() - self.start):0.6f} s')

# Read input data and deserialize it, also has error handling
try:
    with open("data/input/input_network_data.json") as fp:
        try: 
            data = fp.read()
        except:
            raise ValidationException("Error reading input data file")
except:
    raise ValidationException("Error reading input data file")

try:
    input_data = json_deserialize(data)
except:
    raise ValidationException("Error deserializing input data")

#print("components:", list(dataset.keys()))
#display(dataset[ComponentType.line]["from_node"])
#display(DataFrame(dataset[ComponentType.line]))

# Read the batch data data and transform it into a DataFrame
active_power_profile = pd.read_parquet("data/input/active_power_profile.parquet")
reactive_power_profile = pd.read_parquet("data/input/reactive_power_profile.parquet")
if not active_power_profile.index.equals(reactive_power_profile.index): # Check if the timestamps are the same else give an error
    raise ValidationException("Active and reactive batch data have different timestamps")
#display(active_power_profile)
#display(reactive_power_profile)

# Run batch series with active power profile (This should probably be the input to the overall function when implemented, then it would also be easier to do it with the reactive profile)
load_profile_active = initialize_array(DatasetType.update, ComponentType.sym_load, active_power_profile.shape)
load_profile_active["id"] = active_power_profile.columns.to_numpy()
load_profile_active["p_specified"] = active_power_profile.to_numpy()
load_profile_active["q_specified"] = 0.0
update_data = {ComponentType.sym_load: load_profile_active}

# Assert valid data and calculate power flow
try:
    assert_valid_batch_data(input_data=input_data, update_data=update_data, calculation_type=CalculationType.power_flow)
    model = PowerGridModel(input_data=input_data)
    output_data = model.calculate_power_flow(update_data=update_data, calculation_method=CalculationMethod.newton_raphson)
except ValidationException as ex:
    for error in ex.errors:
        print(type(error).__name__, error.component, ":", error.ids)


######################################### Implementing the functionalities #########################################


###################### Calculating the maximum and minimum voltage at each node for each timestamp ######################

ids = np.unique(output_data[ComponentType.node]['id']) # Find amout of unique nodes in the update batch data
ids_node = active_power_profile.index

min_voltage = np.zeros(len(ids_node))
max_voltage = np.zeros(len(ids_node))
min_voltage_id = np.zeros(len(ids_node), dtype=object)
max_voltage_id = np.zeros(len(ids_node), dtype=object)
# display(output_data[ComponentType.node].dtype.names)
# display(output_data[ComponentType.node]["u_pu"])
# display(output_data[ComponentType.node]["u_pu"][0,:].min())

for i in range(len(ids_node)):
    min_voltage[i] = output_data[ComponentType.node]["u_pu"][i,:].min()
    min_voltage_id[i] = output_data[ComponentType.node]["u_pu"][i,:].argmin()+1
    max_voltage[i] = output_data[ComponentType.node]["u_pu"][i,:].max()
    max_voltage_id[i] = output_data[ComponentType.node]["u_pu"][i,:].argmax()+1

output_node = pd.DataFrame()
output_node["id"] = ids_node
output_node["Max_voltage"] = max_voltage
output_node["Max_voltage_node"] = max_voltage_id
output_node["Min_voltage"] = min_voltage
output_node["Min_voltage_node"] = min_voltage_id
print("\nSelf-made output node table: ")
display(output_node)
################## Calculating the maximum and minimum loading for each line and total energy loss ##################

# display(output_data[ComponentType.line].dtype.names)
ids = np.unique(output_data[ComponentType.line]['id']) # Find amout of unique loads in the update batch data
# print(len(ids))

min_loading = np.zeros(len(ids))
max_loading = np.zeros(len(ids))
min_loading_timestamp = np.zeros(len(ids), dtype=object)
max_loading_timestamp = np.zeros(len(ids), dtype=object)
for i in range(len(ids)):
    min_loading[i] = output_data[ComponentType.line]["loading"][:,i].min()
    for j in range(len(active_power_profile.index)): # Find the timestamp of the minimum loading
        if min_loading[i] == output_data[ComponentType.line]["loading"][j,i]:
            min_loading_timestamp[i] = active_power_profile.index[j]

    max_loading[i] = output_data[ComponentType.line]["loading"][:,i].max()
    for j in range(len(active_power_profile.index)): # Find the timestamp of the maximim loading
        if max_loading[i] == output_data[ComponentType.line]["loading"][j,i]:
            max_loading_timestamp[i] = active_power_profile.index[j]

    # Add the energy loss with the trapezoidal rule?

# display(output_data[ComponentType.line]["loading"])
# display(max_loading)
# display(max_loading_timestamp)

output_line = pd.DataFrame()
output_line["id"] = ids
output_line["Total_loss"] = np.zeros(len(ids)) # Not yet calculated
output_line["Max_loading"] = max_loading
output_line["Max_loading_timestamp"] = max_loading_timestamp
output_line["Min_loading"] = min_loading
output_line["Min_loading_timestamp"] = min_loading_timestamp
print("\nSelf-made line table: ")
display(output_line)

# Testing if processed data is the same as the expected data
# display(output_data[ComponentType.sym_load])

expected_output_timestamp = pd.read_parquet("data\expected_output\output_table_row_per_timestamp.parquet")
expected_output_line = pd.read_parquet("data\expected_output\output_table_row_per_line.parquet")

print("\nProvided timestamp table: ")
display(expected_output_timestamp)
print("\nProvided line table: ")
display(expected_output_line)
