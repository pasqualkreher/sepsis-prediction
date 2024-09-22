import pandas as pd
import warnings
warnings.simplefilter('ignore')

def preselect_data():

    excluding_list = []

    ###########################################################################
    ### 1. Exclude ICUSTAYS with over 50% missing data by chart and lab data
    ###########################################################################
    # Read chart data
    df_chartevents_time_shaped   = pd.read_csv(r"./data/temp_data/df_chartevents_time_shaped.csv", index_col=0)
    df_labevents_time_shaped     = pd.read_csv(r"./data/temp_data/df_labevents_time_shaped.csv", index_col=0)

    # Here we identify ICUSTAYS with more than 50% missing columns for removal.
    # We use the maximum value for each ICUSTAY, so if at least one record exists for the entire stay,
    # we count it as available. Its applied for Chart- and Labevents
    df_chartevents_time_max   = df_chartevents_time_shaped.groupby("ICUSTAY_ID").max()
    df_chartevents_time_na    = df_chartevents_time_max.isna().mean(axis=1)
    chartevents_exclude       = df_chartevents_time_na[df_chartevents_time_na > 0.50].index.to_list() # Index are the ICUSTAYS

    df_labevents_time_max     = df_labevents_time_shaped.groupby("ICUSTAY_ID").max()
    df_labevents_time_na      = df_labevents_time_max.isna().mean(axis=1)
    labevents_exclude         = df_labevents_time_na[df_labevents_time_na > 0.50].index.to_list() # Index are the ICUSTAYS

    excluding_list_missing = chartevents_exclude + labevents_exclude
    excluding_list.extend(excluding_list_missing) # Saving the ICUSTAYS for exclusion in the list 'excluding_list' row 5
    del df_chartevents_time_shaped
    del df_labevents_time_shaped
    
    ###########################################################################
    ### 2. Exclude ICUSTAYS
    ###########################################################################
    # Exclude Sepsis like admission diagnosis
    # Read Subject data
    df_subjects                     = pd.read_csv(r"./data/extracted_data/subjects.csv", index_col=0)
    # Pick ICUSTAY IDs with Sepsis admission diagnosis for removal
    mask                            = df_subjects["DIAGNOSIS"].str.upper().str.contains("SEPSIS|SEPTIC") == True
    excluding_list_spsis_admission  = df_subjects[mask]["ICUSTAY_ID"].unique().tolist()
    excluding_list.extend(excluding_list_spsis_admission) # Saving the ICUSTAYS for exclusion in the list 'excluding_list' row 5

    # Exclude age <= 14
    mask                = df_subjects["AGE"] <= 14
    excluding_list_age  = df_subjects[mask]["ICUSTAY_ID"].unique().tolist()
    excluding_list.extend(excluding_list_age) # Saving the ICUSTAYS for exclusion in the list 'excluding_list' row 5

    # Exclude multiple admission and keep first icustay
    first_icustay                   = df_subjects.sort_values(by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], ascending=True)
    first_icustay                   .drop_duplicates(subset=["SUBJECT_ID"], keep="first", inplace=True)
    first_icustay_ids               = first_icustay["ICUSTAY_ID"].to_list()
    mask                            = df_subjects["ICUSTAY_ID"].isin(first_icustay_ids) == True
    excluding_list_multiple_icustay = df_subjects[~mask]["ICUSTAY_ID"].to_list()
    excluding_list.extend(excluding_list_multiple_icustay) # Saving the ICUSTAYS for exclusion in the list 'excluding_list' row 5

    ###########################################################################
    ### 3. Exclude ICUSTAYS
    ###########################################################################
    reduce_df= {
        "df_subjects_reduced"           : r"./data/extracted_data/subjects.csv",
        "df_time_ranges_reduced"        : r"./data/temp_data/df_time_ranges_shaped.csv",
        "df_labevents_time_reduced"     : r"./data/temp_data/df_labevents_time_shaped.csv",
        "df_vent_time_reduced"          : r"./data/temp_data/df_vent_time_shaped.csv",
        "df_input_time_reduced"         : r"./data/temp_data/df_input_time_shaped.csv",
        "df_chartevents_time_reduced"   : r"./data/temp_data/df_chartevents_time_shaped.csv",
        "df_output_time_reduced"        : r"./data/temp_data/df_output_time_shaped.csv",
        "df_culture_shaped_reduced"     : r"./data/temp_data/df_culture_shaped.csv",
        "df_susp_reduced"               : r"./data/temp_data/df_susp_shaped.csv"
    }

    # Apply the excluding list to all dataframes
    for key in reduce_df:
        df = pd.read_csv(reduce_df[key], index_col=0)
        df = df[~(df["ICUSTAY_ID"].isin(excluding_list))]
        df .to_csv(fr"./data/temp_data/{key}.csv")
        del df

def main():
    preselect_data()

if __name__ == '__main__':
    main()
