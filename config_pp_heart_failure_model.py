# ====   PATHS ===================

PATH_TO_DATASET = "C:/Users/amaur/OneDrive/Documents/Data science/Decease of hart fail prediction/deploy_heart_failure_model/heart_failure_mortality.csv"
OUTPUT_PATH = "C:/Users/amaur/OneDrive/Documents/Data science/Decease of hart fail prediction/deploy_heart_failure_model/heart_logit_sk.pkl"
OUTPUT_PATH_SCALER = "C:/Users/amaur/OneDrive/Documents/Data science/Decease of hart fail prediction/deploy_heart_failure_model/scaler.pkl"

# ===    Var declaration ===========

remove_var = ['']

cleanup_types = {"Smoker": {"non smoker": 0, "smoker": 1},"anaemic": {"yes": 1, "no": 0},
               "high_blood_pression": {"yes": 1, "no": 0}, "diabetic": {"yes": 1, "no": 0},
               "gender": {"male": 1, "female": 0}}


var_continuous = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

test_var = ['age', 'creatinine_phosphokinase']

var_dep = ['DEATH_EVENT']

cat_var = ['Smoker', 'anaemic', 'high_blood_pression', 'diabetic', 'gender']

feature = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'Smoker', 'anaemic', 'high_blood_pression', 'diabetic', 'gender']

# ===  Constant =======

max_quant = 0.80
min_quant = 0.20