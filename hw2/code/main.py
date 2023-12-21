import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
import numpy as np
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import argparse
def preprocess(data_file):
    original_train_data = pd.read_csv(data_file)
    
    numeric_columns = original_train_data.select_dtypes(include='number').columns
    # interpolate the missing values
    for column in numeric_columns:

        original_train_data[column] = original_train_data[column].interpolate(method='linear', limit_direction='both')

    for column in original_train_data.columns:

        if original_train_data[column].isnull().any():

            value_counts = original_train_data[column].value_counts()

            most_frequent_value = value_counts.idxmax()

            original_train_data[column] = original_train_data[column].fillna(most_frequent_value)
        
    #use one-hot encoding for categorical features
    original_train_data = pd.get_dummies(original_train_data)
    #use standardization for numeric features

    numeric_columns = original_train_data.select_dtypes(include='number').columns
    scaler = StandardScaler()
    original_train_data[numeric_columns] = scaler.fit_transform(original_train_data[numeric_columns])
    original_train_data = pd.get_dummies(original_train_data).astype(float)
    selected_features = ['hospital_id', 'age', 'bmi', 'height', 'icu_id', 'pre_icu_los_days',
       'weight', 'apache_2_diagnosis', 'apache_3j_diagnosis',
       'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache',
       'heart_rate_apache', 'intubated_apache', 'map_apache',
       'resprate_apache', 'temp_apache', 'ventilated_apache', 'd1_diasbp_max',
       'd1_diasbp_min', 'd1_diasbp_noninvasive_max',
       'd1_diasbp_noninvasive_min', 'd1_heartrate_max', 'd1_heartrate_min',
       'd1_mbp_max', 'd1_mbp_min', 'd1_mbp_noninvasive_max',
       'd1_mbp_noninvasive_min', 'd1_resprate_max', 'd1_resprate_min',
       'd1_spo2_max', 'd1_spo2_min', 'd1_sysbp_max', 'd1_sysbp_min',
       'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min', 'd1_temp_max',
       'd1_temp_min', 'h1_diasbp_max', 'h1_diasbp_min',
       'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min',
       'h1_heartrate_max', 'h1_heartrate_min', 'h1_mbp_max', 'h1_mbp_min',
       'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min', 'h1_resprate_max',
       'h1_resprate_min', 'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_max',
       'h1_sysbp_min', 'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min',
       'd1_glucose_max', 'd1_glucose_min', 'd1_potassium_max',
       'd1_potassium_min', 'apache_4a_hospital_death_prob',
       'apache_4a_icu_death_prob']
    selected_train_X = original_train_data[selected_features]
    return selected_train_X
    
if __name__ == '__main__':   
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)   
    parser = argparse.ArgumentParser(description='Data mining hw2')

    parser.add_argument('--input_file', '-i', type=str, default='test_X.csv', help='Input file path')
    parser.add_argument('--output_file', '-o', type=str, default='test_pred.csv', help='Output file path')
    parser.add_argument('--submission_file', '-s', type=str, default='sample_submission.csv', help='Sample submission file path')
    args = parser.parse_args()
    print("Start testing...")
    test_data = preprocess(args.input_file)
    sample_csv = pd.read_csv(args.submission_file)
    loaded_model = XGBClassifier()
    loaded_model.load_model('model.json')

    test_data = test_data
    test_pred_prob = loaded_model.predict_proba(test_data)[:, 1]
    test_pred = (test_pred_prob > 0.27500000000000013).astype(int)
    test_pred = pd.DataFrame(test_pred)
    test_pred.columns = ['has_died']
    #combine test_pred and sample_csv
    sample_csv["pred"] = test_pred["has_died"]
    # print(sample_csv)
    sample_csv.to_csv(args.output_file, index=False)
    print("Finish testing...")
    print(f"Save prediction to {args.output_file}")