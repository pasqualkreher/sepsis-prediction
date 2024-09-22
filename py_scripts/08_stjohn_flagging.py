import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

def stjohn_flagging():
    ###########################################################################
    ### Sequentially apply function for .groupby("ICUSTAY_ID")
    ###########################################################################
    def apply_sequential(df_grouped, func, desc=None):
        groups = [group for _, group in df_grouped]
        results = [func(group) for group in tqdm(groups, total=len(groups), desc=desc)]
        return pd.concat(results)

    ###########################################################################
    ### 1. Prepare the data for calculating St. John flag
    ###########################################################################
    # Reading relevant dataframes
    df_chartevents_time = pd.read_csv(r"./data/temp_data/df_chartevents_time_filled.csv", index_col=0)
    df_labevents_time   = pd.read_csv(r"./data/temp_data/df_labevents_time_filled.csv", index_col=0)
    df_time_ranges      = pd.read_csv(r"./data/temp_data/df_time_ranges_reduced.csv", index_col=0)
     
    # Reduce some dataframes to only relevant columns for the Sepsis Onset calculation
    a = df_chartevents_time[["CHARTTIME_FLOOR", "HADM_ID", "ICUSTAY_ID", "TEMPC", "HEARTRATE", "RESPRATE", "MEANBP", "GLUCOSE"]]
    del df_chartevents_time
    b = df_labevents_time[["CHARTTIME_FLOOR", "HADM_ID", "ICUSTAY_ID", "WBC", "LACTATE", "BILIRUBIN", "CREATININE"]]
    del df_labevents_time

    # Merging the data to one dataframe
    df_stjohn = df_time_ranges.copy()
    for df in [a, b]:
        df_stjohn = df_stjohn.merge(df, how="left")
        del df
    del df_time_ranges

    df_stjohn["CHARTTIME_FLOOR"] = pd.to_datetime(df_stjohn["CHARTTIME_FLOOR"])

    # Make function with St. John Rules
    def calculate_st_john(group):
        group.sort_values(by=["ICUSTAY_ID", "CHARTTIME_FLOOR"])

        # Temperature
        group["TEMPC_d"] = np.where(
            (group["TEMPC"] < 36) | 
            (group["TEMPC"] > 38.3), 
            1, 
            0
        )

        # HEARTRATE
        group["HEARTRATE_d"] = np.where(
            (group["HEARTRATE"] > 95), 
            1, 
            0
        )

        # RESPRATE
        group["RESPRATE_d"] = np.where(
            (group["RESPRATE"] >= 22), 
            1, 
            0
        )

        # WHITEBLOODCELL
        group["WHITEBLOODCELL_d"] = np.where(
            (group["WBC"] < 4) |
            (group["WBC"] > 12), 
            1, 
            0
        )

        # Glucose
        group["GLUCOSE_d"] = np.where(
            (group["GLUCOSE"] > 141) &
            (group["GLUCOSE"] < 200), 
            1, 
            0
        )

        # MEANBP
        group["MEANBP_d"] = np.where(
            (group["MEANBP"] < 65),
            1, 
            0
        )

        # Lactate
        group["LACTATE_d"] = np.where(
            (group["LACTATE"] > 2),
            1, 
            0
        )

        # Bilirubin
        group["BILIRUBIN_d"] = np.where(
            (group["BILIRUBIN"] > 2) &
            (group["BILIRUBIN"] < 10), 
            1, 
            0
        )

        # Creatinine
        group["CREATININE_d"] = np.where(
            ((group["CREATININE"] - group["CREATININE"].iloc[0]) >= 0.5),
            1, 
            0
        )

        group.sort_values(by='CHARTTIME_FLOOR', inplace=True)
        group["SEPSIS_ALERT"] = np.where((group[['TEMPC_d', 'HEARTRATE_d', 'RESPRATE_d', 'WHITEBLOODCELL_d', 'GLUCOSE_d']].sum(axis=1)) >= 3, 1, 0)

        temp_map = group.rolling('30h', on='CHARTTIME_FLOOR')[['MEANBP_d']].max()
        temp_lac = group.rolling('12h', on='CHARTTIME_FLOOR')[['LACTATE_d']].max()
        temp_bil = group.rolling('30h', on='CHARTTIME_FLOOR')[['BILIRUBIN_d']].max()
        temp_dre = group.rolling('72h', on='CHARTTIME_FLOOR')[['CREATININE_d']].max()
        group["SEVERE_SEPSIS_ALERT"] = np.where(
            ((group[['TEMPC_d', 'HEARTRATE_d', 'RESPRATE_d', 'WHITEBLOODCELL_d', 'GLUCOSE_d']].sum(axis=1)) >= 2) &\
            ((temp_map["MEANBP_d"] + temp_lac["LACTATE_d"] + temp_bil["BILIRUBIN_d"] + temp_dre["CREATININE_d"]) >= 1), 
            1, 0
        )
        group["SEPSIS_ALERT"]           = group["SEPSIS_ALERT"].cummax()
        group["SEVERE_SEPSIS_ALERT"]    = group["SEVERE_SEPSIS_ALERT"].cummax()
        return group

    df_stjohn_flag_calculation = apply_sequential(df_stjohn.groupby(['ICUSTAY_ID']), calculate_st_john, "Set StJohn Flag").reset_index(drop=True) # Apply the function on each ICUSTAY
    df_stjohn_flag_calculation.to_csv(r"./data/temp_data/df_stjohn_flag_calculation.csv")

    df_stjohn_flag = df_stjohn_flag_calculation[["ICUSTAY_ID", "CHARTTIME_FLOOR", "HOUR", "SEPSIS_ALERT", "SEVERE_SEPSIS_ALERT"]]
    df_stjohn_flag.to_csv(r"./data/target_data/df_stjohn_flag.csv")

def main():
    stjohn_flagging()
    
if __name__ == '__main__':
    main()