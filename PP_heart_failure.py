#########################################
######## Start imports ##################
#########################################

# to handle datasets
import pandas as pd
import numpy as np

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to build the models

from sklearn.linear_model import LogisticRegression

# to persist the model and the scaler
import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.simplefilter(action='ignore')

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

################################################
####### end import #############################
################################################

def load_data(df_path):
	return pd.read_csv(df_path, sep=';')


################################################
############ Divide train and test set #########
################################################

def devide_train_test(df, var_dep):
	X_train, X_test, y_train, y_test = train_test_split(df,
                                                    df[var_dep],
                                                    test_size=0.35,
                                                    # we are setting the seed here:
                                                    random_state=0)
	return X_train, X_test, y_train, y_test

################################################
######### Process data on global dataset #######
################################################

def remove_useless_data(df, remove_var):
	for var in remove_var:
		df.drop(var, axis=1, inplace=True)
	return df

###########################################
###### Trim outlier by discarding the max #
###### and min quantiles ##################

def trimming_data(df, max_quant, min_quant):
    Qmin = df.quantile(min_quant)
    Qmax = df.quantile(max_quant)
    IQR = Qmax - Qmin
    
    df = df[~((df < (Qmin - 1.5 * IQR)) |(df > (Qmax + 1.5 * IQR))).any(axis=1)]
    return df

#############################################
### Turn data to log (will be applied only ##
###on continuous skewed variables)###########
#############################################


def log_transform(df, var):
    # apply logarithm transformation to variable
    return np.log(df[var])

### Scaler time ###############################

def train_scaler(df, output_path):
    scaler = MinMaxScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler

def scale_features(df, scaler):
    scaler = joblib.load(scaler) # with joblib probably
    return scaler.transform(df)


###############################################
#### Encode 0 or 1 (only on categorical var) ##
###############################################

def encode_dep(df, cleanup_types):
    df.replace(cleanup_types, inplace=True)
    return df

###########################################
#### Train model ##########################
###########################################

def train_model(df, var_dep, output_path):
	log_reg = LogisticRegression()
	log_reg.fit(df, var_dep)
	joblib.dump(log_reg, output_path)
	return log_reg

##########################################
### Predict outcome ######################
##########################################

def predict(df, model):
    model = joblib.load(model)
    return model.predict(df)
