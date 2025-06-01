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


class MoreThanOneTransformerOrSource(Exception):
    "Raised when there is more than one transformer or source in meta_data.json"

def check_source_transformer(meta_data):
    if((type(meta_data['source']) is not int) or (type(meta_data['transformer'])is not int)):
        raise MoreThanOneTransformerOrSource("LV grid contains more than one source or transformer")
    
def input_data_validity_check(input_data,meta_data):
    
    assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow)
    check_source_transformer(meta_data)
    print(type(input_data))
    print(input_data)

with open("data/assignment 3 input/input_network_data.json") as fp:
    data = fp.read()

input_data = json_deserialize(data)

#print("components:", list(input_data.keys()))
#display(input_data[ComponentType.node])
#display(pd.DataFrame(input_data[ComponentType.node]))
#print(pd.DataFrame(input_data[ComponentType.transformer]))


with open("data/assignment 3 input/meta_data.json") as fp:
    meta = fp.read()

#pprint.pprint(json.load(meta))

meta_data=json.loads(meta)
#pprint.pprint(meta_data['lv_feeders'])

input_data_validity_check(input_data,meta_data)

pass
