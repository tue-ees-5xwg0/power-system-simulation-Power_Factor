import time
from typing import Dict
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from IPython.display import display
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
    dataset = json_deserialize(data)
except:
    raise ValidationException("Error deserializing input data")

print("components:", list(dataset.keys()))
#display(dataset[ComponentType.line]["from_node"])
#display(DataFrame(dataset[ComponentType.line]))

# Read the batch data data and transform it into a DataFrame
active_power_batch = pd.read_parquet("data/input/active_power_profile.parquet")
reactive_power_batch = pd.read_parquet("data/input/reactive_power_profile.parquet")
if not active_power_batch.index.equals(reactive_power_batch.index): # Check if the timestamps are the same else give an error
    raise ValidationException("Active and reactive batch data have different timestamps")
