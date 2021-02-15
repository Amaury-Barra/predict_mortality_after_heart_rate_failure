import numpy as np

import PP_heart_failure as phf
import config_pp_heart_failure_model as config
import warnings
warnings.simplefilter(action='ignore')

######################################
############ Load data ###############
######################################

data = phf.load_data(config.PATH_TO_DATASET)

######################################
############ Transform data ##########
######################################

## Dataset already clean, otherwise should have done ########
#data = phf.remove_useless_data(data, config.remove_var)

data = phf.trimming_data(data, config.max_quant, config.min_quant)


for var in data[config.var_continuous]:
	data[var] = phf.log_transform(data, var)


data = phf.encode_dep(data, config.cleanup_types)

######################################
############ Scale data ##############
######################################

scaler = phf.train_scaler(data[config.var_continuous], config.OUTPUT_PATH_SCALER)

scaler.transform(data[config.var_continuous])

####################################################
########## Splitting dataset #######################
####################################################

X_train, X_test, y_train, y_test = phf.devide_train_test(data, config.var_dep)

####################################################
########## Training selected model #################
####################################################

phf.train_model(X_train[config.feature], y_train, config.OUTPUT_PATH)

print('finished training')