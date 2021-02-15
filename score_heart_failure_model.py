import PP_heart_failure as phf
import config_pp_heart_failure_model as config

def predict(data):
    ## Preprocessing
    ## Predicting
    predictions = phf.predict(data, config.OUTPUT_PATH)
    return predictions

#########################################
##### Test that script is going OK ######
#########################################

if __name__ == '__main__':
    
    from math import sqrt
    import numpy as np
    
    from sklearn.metrics import confusion_matrix
    
    import warnings
    warnings.simplefilter(action='ignore')

    data = phf.load_data(config.PATH_TO_DATASET)

    data = phf.trimming_data(data, config.max_quant, config.min_quant)

    for var in data[config.var_continuous]:
    	data[var] = phf.log_transform(data, var)

    data = phf.encode_dep(data, config.cleanup_types)

    scaler = phf.train_scaler(data[config.var_continuous], config.OUTPUT_PATH_SCALER)

    scaler.transform(data[config.var_continuous])

    X_train, X_test, y_train, y_test = phf.devide_train_test(data, config.var_dep)

    X_test = X_test[config.feature]

    pred = predict(X_test)

    print(len(y_test))
    print(pred)
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, pred)

    print(confusion_matrix)