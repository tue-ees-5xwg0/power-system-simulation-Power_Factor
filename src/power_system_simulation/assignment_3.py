# some basic imports
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
import pprint
import json

with open("data/assignment 3 input/input_network_data.json") as fp:
    data = fp.read()

#pprint.pprint(json.loads(data))

input_data = json_deserialize(data)
assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow)

#print("components:", list(input_data.keys()))
#display(input_data[ComponentType.node])
#display(pd.DataFrame(input_data[ComponentType.node]))
#print(pd.DataFrame(input_data[ComponentType.transformer]))


with open("data/assignment 3 input/meta_data.json") as fp:
    meta = fp.read()

pprint.pprint(json.loads(meta))

#meta_data = json_deserialize(meta)
#assert_valid_input_data(input_data=meta_data, calculation_type=CalculationType.power_flow)

#print("components:", list(meta_data.keys()))
#display(meta_data[ComponentType.node])
#display(pd.DataFrame(meta_data[ComponentType.node]))
#print(pd.DataFrame(meta_data[ComponentType.transformer]))