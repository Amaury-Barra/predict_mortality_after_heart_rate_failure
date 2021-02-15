"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.
This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import numpy as np
import pandas as pd


## Homemade folders
import config_pp_heart_failure_model as config

# lr_model is our simple logistic regression model
# lr_model.feature_names are the four different iris measurements

with open("C:/Users/amaur/OneDrive/Documents/Data science/Decease of hart fail prediction/deploy_heart_failure_model/heart_logit_sk.pkl", "rb") as f:
    print(pickle.load(f))

print('OK so far')

