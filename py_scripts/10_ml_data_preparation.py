import pandas as pd
import numpy as np
import random
import warnings
warnings.simplefilter('ignore')

def ml_data_preparation():
    # This function create a total balanced dataframe meant to be used for the
    # Machine-Learning Traing process

    ###########################################################################
    ### 1. Reading the data 
    ###########################################################################
    df_chartevents  = pd.read_csv(r"./data/target_data/df_chartevents_time.csv", index_col=0)
    df_labevents    = pd.read_csv(r"./data/target_data/df_labevents_time.csv", index_col=0)
    df_output       = pd.read_csv(r"./data/target_data/df_output_time.csv", index_col=0)
    df_sepsis_flag  = pd.read_csv(r"./data/target_data/df_sepsis_flag.csv", index_col=0)
    df_subjects     = pd.read_csv(r"./data/target_data/df_subjects.csv", index_col=0)
    df_vent         = pd.read_csv(r"./data/target_data/df_vent_time.csv", index_col=0)
    # df_sofa         = pd.read_csv(r"./data/temp_data/df_sepsis_flag_calculation.csv", index_col=0)
    # df_sofa         = df_sofa[["CHARTTIME_FLOOR", "ICUSTAY_ID", "HADM_ID", "SOFA_SCORE", "HOUR"]]

    ###########################################################################
    ### 3. Mergeing the dataframes 
    ###########################################################################
    df = df_chartevents.iloc[:,:4]
    df = df.merge(df_sepsis_flag, how="left", on=["ICUSTAY_ID", "CHARTTIME_FLOOR", "HOUR"])
    df = df.merge(df_chartevents, how="left", on=["ICUSTAY_ID", "HADM_ID", "CHARTTIME_FLOOR", "HOUR"])
    df = df.merge(df_labevents, how="left", on=["ICUSTAY_ID", "HADM_ID", "CHARTTIME_FLOOR", "HOUR"])
    df = df.merge(df_output, how="left", on=["ICUSTAY_ID", "HADM_ID", "CHARTTIME_FLOOR", "HOUR"])
    df = df.merge(df_vent, how="left", on=["ICUSTAY_ID", "HADM_ID","CHARTTIME_FLOOR", "HOUR"])
    # df = df.merge(df_sofa, how='left', on=["ICUSTAY_ID", "HADM_ID", "CHARTTIME_FLOOR", "HOUR"])  
    df = df.merge(df_subjects[["HADM_ID", "ICUSTAY_ID", "AGE"]], how="inner" , on=["HADM_ID", "ICUSTAY_ID"])    
    
    del df_chartevents
    del df_labevents
    del df_sepsis_flag
    del df_vent
    del df_subjects

    ###########################################################################
    ### 4. Find a no sepsis case for each sepsis case
    ###########################################################################

    ### 4.1 Create List with possible no Sepsis ICUSTAY IDs matched on Sepsis IDs
    random.seed(42)
    sepsis_onset_hour   = df[df["SEPSIS_FLAG"] == 1].groupby("ICUSTAY_ID")[["HOUR"]].min().rename(columns={"HOUR": "SEPSIS_ONSET_HOUR"})
    max_hour            = df.groupby("ICUSTAY_ID")[["HOUR"]].max().rename(columns={"HOUR": "MAX_HOUR"})
    sepsis_flag         = df.groupby("ICUSTAY_ID")[["SEPSIS_FLAG"]].max()
    df_hour_sepsis_flag = (sepsis_flag.join(sepsis_onset_hour).join(max_hour)).reset_index()

    dict_sepsis_flag = {}
    for hour in sorted(df_hour_sepsis_flag['MAX_HOUR'].unique()):
        no_sepsis_ids = df_hour_sepsis_flag[(df_hour_sepsis_flag['SEPSIS_FLAG'] == 0) &
                                        #(df_hour_sepsis_flag['MAX_HOUR'] == hour)]['ICUSTAY_ID'].unique()
                                        ((df_hour_sepsis_flag['MAX_HOUR'] >= hour) & (df_hour_sepsis_flag['MAX_HOUR'] <= hour + 24))]['ICUSTAY_ID'].unique()
        if no_sepsis_ids.size > 0:
            dict_sepsis_flag[hour] = no_sepsis_ids.tolist()

    df_hour_sepsis_flag["NO_SEPSIS_IDS"]        = df_hour_sepsis_flag[df_hour_sepsis_flag["SEPSIS_FLAG"]==1]["MAX_HOUR"].map(dict_sepsis_flag)
    df_hour_sepsis_flag['RANDOM_NO_SEPSIS_ID']  = None  

    temp_df_hour    = df_hour_sepsis_flag.copy()
    sepsis_ids_mask = (temp_df_hour["SEPSIS_FLAG"] == 1) & temp_df_hour["NO_SEPSIS_IDS"].notna()

    ### 4.2  Match one of the closest no Sepsis IDs on the Spesis IDs
    for sepsis_id in temp_df_hour.loc[sepsis_ids_mask, "ICUSTAY_ID"]:

        sepsis_id_mask  = (temp_df_hour["ICUSTAY_ID"] == sepsis_id)
        max_hour_sepsis = temp_df_hour.loc[sepsis_id_mask, "MAX_HOUR"].item()
        no_sepsis_ids   = temp_df_hour.loc[sepsis_id_mask, "NO_SEPSIS_IDS"].values[0]

        no_sepsis_ids_mask = temp_df_hour["ICUSTAY_ID"].isin(no_sepsis_ids)
        max_hour_no_sepsis = temp_df_hour.loc[no_sepsis_ids_mask, "MAX_HOUR"]
        
        hour_differences = max_hour_no_sepsis - max_hour_sepsis
        hour_differences = hour_differences[hour_differences >= 0]  

        if hour_differences.empty:
            continue

        for i in range(24):
            closest_match_indices = hour_differences[hour_differences == hour_differences.min() + i].index
            
            if not closest_match_indices.empty:
                random_no_sepsis_id = random.choice(temp_df_hour.loc[closest_match_indices, "ICUSTAY_ID"].to_list())

                if random_no_sepsis_id in temp_df_hour["ICUSTAY_ID"].values:
                    temp_df_hour.loc[temp_df_hour["ICUSTAY_ID"] == sepsis_id, "RANDOM_NO_SEPSIS_ID"] = random_no_sepsis_id
                    print(f"Sepsis ID {sepsis_id} paired with non-sepsis ID {random_no_sepsis_id} with time-difference of {hour_differences.min() + i}")

                    temp_df_hour = temp_df_hour.loc[temp_df_hour["ICUSTAY_ID"] != random_no_sepsis_id]
                    break

    temp_df_hour                .dropna(subset="RANDOM_NO_SEPSIS_ID", inplace=True)
    temp_df_hour["SEPSIS_ID"]   = temp_df_hour["ICUSTAY_ID"] 
    df_hour_temp                = temp_df_hour[["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID", "MAX_HOUR" ,"SEPSIS_ONSET_HOUR"]]
    df_icuids_mapped            = df_hour_temp.astype(int)

    ###########################################################################
    ### 5. Merging the matched ICUSTAY IDs on the df
    ###########################################################################
    df_sepsis       = df.merge(df_icuids_mapped, left_on=["ICUSTAY_ID"], right_on=["SEPSIS_ID"], how="inner")
    df_no_sepsis    = df.merge(df_icuids_mapped, left_on=["ICUSTAY_ID"], right_on=["RANDOM_NO_SEPSIS_ID"], how="inner")
    df_reduce       = pd.concat([df_sepsis, df_no_sepsis], axis=0).reset_index(drop=True)
    del df
    del df_sepsis
    del df_no_sepsis

    ###########################################################################
    ### 6. Shorten the dataframe by smaller than 5h after Sepsis Onset or 
    ###    fictive Sepsis Onset, set sepsis labels and order columns
    ###########################################################################
    df_reduce["SEPSIS_LABEL"]   = np.where(df_reduce["SEPSIS_ID"] == df_reduce["ICUSTAY_ID"], 1, 0)
    df_reduced                  = df_reduce[(df_reduce["HOUR"] < df_reduce["SEPSIS_ONSET_HOUR"] + 5)]
    del df_reduce
    new_col_order               = df_reduced.columns.insert(5, df_reduced.columns[-5:])[:-5]
    df_ml_reduced               = df_reduced[new_col_order]

    ###########################################################################
    ### 7. Fromatting cols
    ###########################################################################
    df_ml_reduced = df_ml_reduced.round(2)
    df_ml_reduced[["MECHVENT"]] = df_ml_reduced[["MECHVENT"]].astype(np.int32)
    df_ml_reduced[["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"]] = df_ml_reduced[["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"]].astype(np.int32)
    for col in df_ml_reduced.columns:
        if "GCS" in col or "FIO2" in col:
            df_ml_reduced[col] = df_ml_reduced[col].round(0)

    ###########################################################################
    ### 8. Saving the dataframes
    ###########################################################################
    df_ml_reduced       .to_csv(r"./data/temp_data/df_ml_reduced.csv")
    df_icuids_mapped    .to_csv(r"./data/target_data/df_icuids_mapped.csv")

def ml_testing_preparation():
    # This function create a unbalanced dataframe meant to be used for the
    # Testing the ML-Models, like realworld condition with unbalanced data

    ###########################################################################
    ### 1. Reading the data 
    ###########################################################################
    df_chartevents  = pd.read_csv(r"./data/target_data/df_chartevents_time.csv", index_col=0)
    df_labevents    = pd.read_csv(r"./data/target_data/df_labevents_time.csv", index_col=0)
    df_output       = pd.read_csv(r"./data/target_data/df_output_time.csv", index_col=0)
    df_sepsis_flag  = pd.read_csv(r"./data/target_data/df_sepsis_flag.csv", index_col=0)
    df_subjects     = pd.read_csv(r"./data/target_data/df_subjects.csv", index_col=0)
    df_vent         = pd.read_csv(r"./data/target_data/df_vent_time.csv", index_col=0)
    # df_sofa         = pd.read_csv(r"./data/temp_data/df_sepsis_flag_calculation.csv", index_col=0)
    # df_sofa         = df_sofa[["CHARTTIME_FLOOR", "ICUSTAY_ID", "HADM_ID", "SOFA_SCORE", "HOUR"]]

    ###########################################################################
    ### 3. Mergeing the dataframes 
    ###########################################################################
    df = df_chartevents.iloc[:,:4]
    df = df.merge(df_sepsis_flag, how="left", on=["ICUSTAY_ID", "CHARTTIME_FLOOR", "HOUR"])
    df = df.merge(df_chartevents, how="left", on=["ICUSTAY_ID", "HADM_ID", "CHARTTIME_FLOOR", "HOUR"])
    df = df.merge(df_labevents, how="left", on=["ICUSTAY_ID", "HADM_ID", "CHARTTIME_FLOOR", "HOUR"])
    df = df.merge(df_output, how="left", on=["ICUSTAY_ID", "HADM_ID", "CHARTTIME_FLOOR", "HOUR"])
    df = df.merge(df_vent, how="left", on=["ICUSTAY_ID", "HADM_ID","CHARTTIME_FLOOR", "HOUR"])
    # df = df.merge(df_sofa, how='left', on=["ICUSTAY_ID", "HADM_ID", "CHARTTIME_FLOOR", "HOUR"])  
    df = df.merge(df_subjects[["HADM_ID", "ICUSTAY_ID", "AGE"]], how="inner" , on=["HADM_ID", "ICUSTAY_ID"])    

    del df_chartevents
    del df_labevents
    del df_sepsis_flag
    del df_vent
    del df_subjects

    ###########################################################################
    ### 4. Find a no sepsis case for each sepsis case
    ###########################################################################

    ### 4.1 Create List with possible no Sepsis ICUSTAY IDs matched on Sepsis IDs
    random.seed(42)
    sepsis_onset_hour   = df[df["SEPSIS_FLAG"] == 1].groupby("ICUSTAY_ID")[["HOUR"]].min().rename(columns={"HOUR": "SEPSIS_ONSET_HOUR"})
    max_hour            = df.groupby("ICUSTAY_ID")[["HOUR"]].max().rename(columns={"HOUR": "MAX_HOUR"})
    sepsis_flag         = df.groupby("ICUSTAY_ID")[["SEPSIS_FLAG"]].max()
    df_hour_sepsis_flag = (sepsis_flag.join(sepsis_onset_hour).join(max_hour)).reset_index()

    dict_sepsis_flag = {}
    for hour in sorted(df_hour_sepsis_flag['MAX_HOUR'].unique()):
        no_sepsis_ids = df_hour_sepsis_flag[(df_hour_sepsis_flag['SEPSIS_FLAG'] == 0) &
                                        #(df_hour_sepsis_flag['MAX_HOUR'] == hour)]['ICUSTAY_ID'].unique()
                                        ((df_hour_sepsis_flag['MAX_HOUR'] >= hour) & (df_hour_sepsis_flag['MAX_HOUR'] <= hour + 24))]['ICUSTAY_ID'].unique()
        if no_sepsis_ids.size > 0:
            dict_sepsis_flag[hour] = no_sepsis_ids.tolist()

    df_hour_sepsis_flag["NO_SEPSIS_IDS"]        = df_hour_sepsis_flag[df_hour_sepsis_flag["SEPSIS_FLAG"]==1]["MAX_HOUR"].map(dict_sepsis_flag)
    df_hour_sepsis_flag['RANDOM_NO_SEPSIS_ID']  = None  

    temp_df_hour    = df_hour_sepsis_flag.copy()
    sepsis_ids_mask = (temp_df_hour["SEPSIS_FLAG"] == 1) & temp_df_hour["NO_SEPSIS_IDS"].notna()

    ### 4.2  Match one of the closest no Sepsis IDs on the Spesis IDs
    for sepsis_id in temp_df_hour.loc[sepsis_ids_mask, "ICUSTAY_ID"]:

        sepsis_id_mask  = (temp_df_hour["ICUSTAY_ID"] == sepsis_id)
        max_hour_sepsis = temp_df_hour.loc[sepsis_id_mask, "MAX_HOUR"].item()
        no_sepsis_ids   = temp_df_hour.loc[sepsis_id_mask, "NO_SEPSIS_IDS"].values[0]

        no_sepsis_ids_mask = temp_df_hour["ICUSTAY_ID"].isin(no_sepsis_ids)
        max_hour_no_sepsis = temp_df_hour.loc[no_sepsis_ids_mask, "MAX_HOUR"]
        
        hour_differences = max_hour_no_sepsis - max_hour_sepsis
        hour_differences = hour_differences[hour_differences >= 0]  

        if hour_differences.empty:
            continue

        for i in range(24):
            closest_match_indices = hour_differences[hour_differences == hour_differences.min() + i].index
            
            if not closest_match_indices.empty:
                random_no_sepsis_id = random.choice(temp_df_hour.loc[closest_match_indices, "ICUSTAY_ID"].to_list())

                if random_no_sepsis_id in temp_df_hour["ICUSTAY_ID"].values:
                    temp_df_hour.loc[temp_df_hour["ICUSTAY_ID"] == sepsis_id, "RANDOM_NO_SEPSIS_ID"] = random_no_sepsis_id
                    print(f"Sepsis ID {sepsis_id} paired with non-sepsis ID {random_no_sepsis_id} with time-difference of {hour_differences.min() + i}")

                    temp_df_hour = temp_df_hour.loc[temp_df_hour["ICUSTAY_ID"] != random_no_sepsis_id]
                    break

    temp_df_hour                .dropna(subset="RANDOM_NO_SEPSIS_ID", inplace=True)
    temp_df_hour["SEPSIS_ID"]   = temp_df_hour["ICUSTAY_ID"] 
    df_hour_temp                = temp_df_hour[["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID", "MAX_HOUR" ,"SEPSIS_ONSET_HOUR"]]
    df_icuids_mapped            = df_hour_temp.astype(int)

    ###########################################################################
    ### 5. Merging the matched ICUSTAY IDs on the df
    ###########################################################################
    df_sepsis       = df.merge(df_icuids_mapped, left_on=["ICUSTAY_ID"], right_on=["SEPSIS_ID"], how="inner")
    df_no_sepsis    = df.merge(df_icuids_mapped, left_on=["ICUSTAY_ID"], right_on=["RANDOM_NO_SEPSIS_ID"], how="inner")
    df_full         = pd.concat([df_sepsis, df_no_sepsis], axis=0).reset_index(drop=True)
    del df_sepsis
    del df_no_sepsis

    # ###########################################################################
    # ### 6. Match fictive Sepsis Onset to no Sepsis cases
    # ###########################################################################
    # Filter and add non-sepsis data
    add_no_sepsis = df[df["ICUSTAY_ID"].isin(df_full["ICUSTAY_ID"]) == False]
    add_no_sepsis = add_no_sepsis.groupby("ICUSTAY_ID").filter(lambda x: x["SEPSIS_FLAG"].max() == 0)
    df_full = pd.concat([df_full.reset_index(drop=True), add_no_sepsis.reset_index(drop=True)], axis=0)

    # Calculate the 75th percentile of SEPSIS_ONSET_HOUR for each MAX_HOUR
    max_hour_sepsis_onset = df.merge(df_hour_sepsis_flag[["ICUSTAY_ID", "MAX_HOUR", "SEPSIS_ONSET_HOUR"]], 
                                    on="ICUSTAY_ID", 
                                    how="left").groupby("MAX_HOUR")["SEPSIS_ONSET_HOUR"].quantile(0.75)

    # Fill missing MAX_HOUR in df_full
    df_full.loc[df_full["MAX_HOUR"].isna(), "MAX_HOUR"] = df_full.loc[df_full["MAX_HOUR"].isna(), "ICUSTAY_ID"].map(
        df_hour_sepsis_flag.set_index("ICUSTAY_ID")["MAX_HOUR"]
    )

    # Fill missing SEPSIS_ONSET_HOUR in df_full based on the calculated quantiles
    df_full.loc[df_full["SEPSIS_ONSET_HOUR"].isna(), "SEPSIS_ONSET_HOUR"] = df_full.loc[df_full["SEPSIS_ONSET_HOUR"].isna(), "MAX_HOUR"].map(
        max_hour_sepsis_onset.to_dict()
    )
    df_full = df_full[df_full["ICUSTAY_ID"].isin(df_full[df_full["SEPSIS_ONSET_HOUR"].isna() == True]["ICUSTAY_ID"].unique()) == False].fillna(99999)
    df_full.reset_index(inplace=True, drop=True)
    
    ###########################################################################
    ### 7. Shorten the dataframe by smaller than 5h after Sepsis Onset or 
    ###    fictive Sepsis Onset, set sepsis labels and order columns
    ###########################################################################
    df_full["SEPSIS_LABEL"]     = np.where(df_full["SEPSIS_ID"] == df_full["ICUSTAY_ID"], 1, 0)
    df_full["SEPSIS_LABEL"]     .fillna(0, inplace=True)
    df_reduced                  = df_full[(df_full["HOUR"] < df_full["SEPSIS_ONSET_HOUR"] + 5)]
    new_col_order               = df_reduced.columns.insert(5, df_reduced.columns[-5:])[:-5]
    df_ml_reduced               = df_reduced[new_col_order]

    ###########################################################################
    ### 8. Fromatting cols
    ###########################################################################
    df_ml_reduced = df_ml_reduced.round(2)
    df_ml_reduced[["MECHVENT"]] = df_ml_reduced[["MECHVENT"]].astype(np.int32)
    df_ml_reduced[["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"]] = df_ml_reduced[["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"]].astype(np.int32)
    for col in df_ml_reduced.columns:
        if "GCS" in col or "FIO2" in col:
            df_ml_reduced[col] = df_ml_reduced[col].round(0)


    ###########################################################################
    ### 9. Saving the dataframes
    ###########################################################################
    df_ml_reduced       .to_csv(r"./data/temp_data/df_ml_testing_reduced.csv")


def main():
    ml_data_preparation()
    ml_testing_preparation()

if __name__ == '__main__':
    main()