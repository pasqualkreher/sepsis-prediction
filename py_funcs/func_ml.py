import pandas as pd
import os
from .func_project_dir import *

class ml_functions(classmethod):
    def read_data():
        df = pd.read_csv(os.path.join(project_path(), "data/target_data/df_ml.csv"), index_col=0)
        return df

    def get_train_test_icustays(df, onset_hour, test_size):
        icustays        = df[(df["SEPSIS_ONSET_HOUR"] > onset_hour)][["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"]]
        icustays        .drop_duplicates(inplace=True)
        test_size       = round(len(icustays) * (1 - test_size))
        df_train_ids    = icustays.sample(n=test_size, random_state=42)
        df_test_ids     = icustays[(icustays["RANDOM_NO_SEPSIS_ID"].isin(df_train_ids["RANDOM_NO_SEPSIS_ID"]) == False) &
                                (icustays["SEPSIS_ID"].isin(df_train_ids["SEPSIS_ID"]) == False)]
        return df_train_ids, df_test_ids

    def make_train_test_ids(onset_hour, test_size):
        df                          = ml_functions.read_data()
        df_train_ids, df_test_ids   = ml_functions.get_train_test_icustays(df, onset_hour=onset_hour, test_size=test_size)
        df_train_ids.to_csv(os.path.join(project_path(), "machine_learning/df_train_ids.csv"))
        df_test_ids.to_csv(os.path.join(project_path(), "machine_learning/df_test_ids.csv"))

    def make_df_for_ml(df, hours_before_sepsis_onset, hours_before_sepsis_cutoff):
        df_ml   = df[(df["HOUR"] >= df["SEPSIS_ONSET_HOUR"] - hours_before_sepsis_onset) &
                     (df["HOUR"] < df["SEPSIS_ONSET_HOUR"]) &
                     (df["HOUR"] < df["SEPSIS_ONSET_HOUR"] - hours_before_sepsis_cutoff)]
        return df_ml
