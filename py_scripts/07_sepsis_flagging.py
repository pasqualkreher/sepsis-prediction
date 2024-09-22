import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

def sepsis_flagging():
    ###########################################################################
    ### Sequentially apply function for .groupby("ICUSTAY_ID")
    ###########################################################################
    def apply_sequential(df_grouped, func, desc=None):
        groups  = [group for _, group in df_grouped]
        results = [func(group) for group in tqdm(groups, total=len(groups), desc=desc)]
        return pd.concat(results)

    ###########################################################################
    ### 1. Prepare the data for calculating SOFA-Scores
    ###########################################################################
    # Reading relevant dataframes
    df_chartevents_time = pd.read_csv(r"./data/temp_data/df_chartevents_time_filled.csv", index_col=0)
    df_labevents_time   = pd.read_csv(r"./data/temp_data/df_labevents_time_filled.csv", index_col=0)
    df_input_time       = pd.read_csv(r"./data/temp_data/df_input_time_imputed.csv", index_col=0)
    df_vent_time        = pd.read_csv(r"./data/temp_data/df_vent_time_imputed.csv", index_col=0)
    df_time_ranges      = pd.read_csv(r"./data/temp_data/df_time_ranges_reduced.csv", index_col=0)
    df_output_time      = pd.read_csv(r"./data/temp_data/df_output_time_reduced.csv", index_col=0)
    df_susp             = pd.read_csv(r"./data/temp_data/df_susp_reduced.csv", index_col=0)

    # Reduce some dataframes to only relevant columns for the Sepsis Onset calculation
    a = df_labevents_time[["CHARTTIME_FLOOR", "HADM_ID", "ICUSTAY_ID", "BILIRUBIN", "CREATININE", "PLATELET", "PO2_BLOODGAS"]]
    del df_labevents_time
    b = df_chartevents_time[["CHARTTIME_FLOOR", "HADM_ID", "ICUSTAY_ID", "GCSEYES", "GCSMOTOR", "GCSVERBAL", "MEANBP", "FIO2"]]
    del df_chartevents_time
    c = df_output_time[["CHARTTIME_FLOOR", "HADM_ID", "ICUSTAY_ID", "URINE_OUPUT_24h"]]
    del df_output_time
    d = df_input_time 
    del df_input_time
    e = df_vent_time 
    del df_vent_time
    

    # Merging the data into one dataframe
    df_sofa = df_time_ranges.copy()
    for df in [a, b, c, d, e]:
        df_sofa = df_sofa.merge(df, how="left")
        del df
    del df_time_ranges
    
    # Calculate Total-GCS Scores and drop the other GCS columns
    df_sofa["GCS"]  = df_sofa["GCSEYES"] + df_sofa["GCSMOTOR"] + df_sofa["GCSVERBAL"]
    df_sofa         .drop(columns=["GCSEYES", "GCSMOTOR", "GCSVERBAL"], inplace=True)
    df_sofa         .sort_values(by=["ICUSTAY_ID", "CHARTTIME_FLOOR"], inplace=True)

    df_sofa["CHARTTIME_FLOOR"]  = pd.to_datetime(df_sofa["CHARTTIME_FLOOR"])
    df_susp["tSUSP"]            = pd.to_datetime(df_susp["tSUSP"])

    ###########################################################################
    ### 2. Calculate Scores for SOFA-Flagging
    ###########################################################################
    # Apply the Sofa-Scoring on the ICUSTAYS
    def calculate_sofa_scores(group):
        group["RESPIRATION_S"] = np.where(((group["PO2_BLOODGAS"] * 100 / group["FIO2"]) < 100) & (group["MECHVENT"] == 1), 4,
                                          np.where(((group["PO2_BLOODGAS"] * 100 / group["FIO2"]) < 200) & (group["MECHVENT"] == 1), 3,
                                          np.where((group["PO2_BLOODGAS"] * 100 / group["FIO2"]) < 300, 2,
                                          np.where((group["PO2_BLOODGAS"] * 100 / group["FIO2"]) < 400, 1,
                                          0))))

        group["COAGULATION_S_"] = np.where(group["PLATELET"] < 20, 4,
                                           np.where(group["PLATELET"] < 50, 3,
                                           np.where(group["PLATELET"] < 100, 2,
                                           np.where(group["PLATELET"] < 150, 1,
                                           0))))

        group["LIVER_S"] = np.where(group["BILIRUBIN"] >= 12, 4,
                                    np.where((group["BILIRUBIN"] >= 6) & (group["BILIRUBIN"] < 12), 3,
                                    np.where((group["BILIRUBIN"] >= 2) & (group["BILIRUBIN"] < 6), 2,
                                    np.where((group["BILIRUBIN"] >= 1.2) & (group["BILIRUBIN"] < 2), 1,
                                    0))))

        group["CARDIOVASCULAR_S"] = np.where(
            (group["RATE_DOPAMINE"] > 15) | (group["RATE_EPINEPHRINE"] > 0.1) | (group["RATE_NOREPINEPHRINE"] > 0.1), 4,
            np.where(
                ((group["RATE_DOPAMINE"] >= 5.1) & (group["RATE_DOPAMINE"] <= 15)) | 
                ((group["RATE_EPINEPHRINE"] > 0) & (group["RATE_EPINEPHRINE"] <= 0.1)) | 
                ((group["RATE_NOREPINEPHRINE"] > 0) & (group["RATE_NOREPINEPHRINE"] <= 0.1)), 3,
            np.where(
                (group["RATE_DOPAMINE"] > 0) & (group["RATE_DOPAMINE"] <= 5) | (group["RATE_DOBUTAMINE"] > 0), 2,
            np.where(
                (group["MEANBP"] < 70), 1,
            0))))

        group["CNS_S"] = np.where(group["GCS"] < 6, 4,
                                  np.where((group["GCS"] >= 6) & (group["GCS"] <= 9), 3,
                                  np.where((group["GCS"] >= 10) & (group["GCS"] <= 12), 2,
                                  np.where((group["GCS"] >= 13) & (group["GCS"] <= 14), 1,
                                  0))))

        group["RENAL_S"] = np.where((group["URINE_OUPUT_24h"] < 200) | (group["CREATININE"] > 5), 4,             
                                    np.where((group["URINE_OUPUT_24h"] < 500) | (group["CREATININE"] >= 3.5), 3,           
                                    np.where((group["CREATININE"] >= 2) & (group["CREATININE"] < 3.5), 2,
                                    np.where((group["CREATININE"] >= 1.2) & (group["CREATININE"] < 2), 1,
                                    0))))

        group                   .sort_values(by='CHARTTIME_FLOOR', inplace=True)
        rolling                 = group.rolling('24h', on='CHARTTIME_FLOOR')
        max_vals                = rolling[['RESPIRATION_S', 'COAGULATION_S_', 'LIVER_S','CARDIOVASCULAR_S', 'CNS_S', 'RENAL_S']].max()    
        group["SOFA_SCORE"] = max_vals[['RESPIRATION_S', 'COAGULATION_S_', 'LIVER_S', 'CARDIOVASCULAR_S', 'CNS_S', 'RENAL_S']].sum(axis=1)        
        return group  

    df_sofa = apply_sequential(df_sofa.groupby("ICUSTAY_ID"), calculate_sofa_scores, "Calculate SOFA-Scores").reset_index(drop=True)
    df_sofa = df_sofa.merge(df_susp)

    ###########################################################################
    ### 3. Calculate Sepsis Onset
    ###########################################################################
    def calculate_tsepsis(group):
        if group["tSUSP"].notna().any() == True:
            group                   .sort_values(by='CHARTTIME_FLOOR', inplace=True)
            sofa_score_first_si     = group[group["CHARTTIME_FLOOR"] >= (group["tSUSP"] - pd.Timedelta(hours=48))]["SOFA_SCORE"].iloc[0]
            group['SOFA_FLAG']      = np.where(group['SOFA_SCORE'] - sofa_score_first_si >= 2, 1, 0)
            group['SOFA_FLAG']      = group['SOFA_FLAG'].cummax()
            condition               = (group["CHARTTIME_FLOOR"] <= (group['tSUSP'] + pd.Timedelta(hours=24))) & \
                                      (group["CHARTTIME_FLOOR"] >= (group['tSUSP'] - pd.Timedelta(hours=48))) & \
                                      (group["SOFA_FLAG"] == 1)
            group["SEPSIS_FLAG"]    = np.where(condition, 1, 0)
            group['SEPSIS_FLAG']    = group['SEPSIS_FLAG'].cummax()
            group["tSEPSIS"]        = np.where(group['SEPSIS_FLAG'].max() == 1, 
                                               pd.to_datetime(group[group["SEPSIS_FLAG"] == 1]["CHARTTIME_FLOOR"].min()), 0)
        else:
            group["SEPSIS_FLAG"]    = 0
            group["tSEPSIS"]        = 0
        return group

    df_sepsis_flag_calculation  = apply_sequential(df_sofa.groupby("ICUSTAY_ID"), calculate_tsepsis, "Set Sepsis Flag").reset_index(drop=True)
    df_sepsis_flag_calculation  .to_csv(r"./data/temp_data/df_sepsis_flag_calculation.csv")
    df_sepsis_flag              = df_sepsis_flag_calculation[["ICUSTAY_ID", "CHARTTIME_FLOOR", "HOUR", "SEPSIS_FLAG"]]
    df_sepsis_flag              .to_csv(r"./data/target_data/df_sepsis_flag.csv")

def main():
    sepsis_flagging()
    
if __name__ == '__main__':
    main()
