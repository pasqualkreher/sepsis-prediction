import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

def data_preprocessing():
    sections = 9  # Total number of sections
    progress_bar = tqdm(total=sections, desc="Progress", position=0)

    ###########################################################################
    ### 1. Creating a time template for each ICUSTAY_ID from subjects.csv
    ###########################################################################
    # Read the subjects.csv
    df_subjects     = pd.read_csv(r"./data/extracted_data/subjects.csv", index_col=0)
    # Remove those which dont have an Intime or an Outtime
    df_subjects.dropna(subset=["INTIME", "OUTTIME"], inplace=True)
    # Function to create a time range from minimum time to maximum time in 1h intervall
    def make_time(df):
        min_time    = df['INTIME'].min()
        max_time    = df['OUTTIME'].max()
        time_index  = pd.date_range(start=min_time, end=max_time, freq='1h').floor('h')
        return pd.DataFrame({'CHARTTIME_FLOOR': time_index, 'ICUSTAY_ID': df['ICUSTAY_ID'].iloc[0]})
    # Apply the funtion to each each ICUSTAY_ID and return the datatime range for each
    df_time_ranges  = df_subjects.groupby('ICUSTAY_ID').apply(lambda group: make_time(group)).reset_index(drop=True)
    # Join the hadm_id again, because lab values dont have ICUSTAY_ID
    df_time_ranges  = df_time_ranges.merge(df_subjects[["HADM_ID", "ICUSTAY_ID"]], how="inner")
    # Create hours for each stayid
    def create_hours(group):
        group["HOUR"] = [i + 1 for i in range(len(group))]
        return group
    df_time_ranges = df_time_ranges.groupby("ICUSTAY_ID").apply(create_hours).reset_index(drop=True)
    # Save df as csv
    df_time_ranges .to_csv(r"./data/temp_data/df_time_ranges_shaped.csv")
    progress_bar.update(1)

    ###########################################################################
    ### 2. Prepare the data from chartevents.csv
    ###########################################################################
    # Read chartevents.csv
    df_chartevents                    = pd.read_csv(r"./data/temp_data/df_chartevents_no_outliers.csv", index_col=0)
    # Checking for non-standard date entries in the 'CHARTTIME' column
    # This checks if each date can be converted to datetime, marking as False if it fails
    mask                        = pd.to_datetime(df_chartevents['CHARTTIME'], errors='coerce').isna()
    df_labevents                      = df_chartevents[~mask]
    # Format charttime to datetime
    df_chartevents["CHARTTIME"]       = pd.to_datetime(df_chartevents["CHARTTIME"])
    # Create charttime_floor and move charttime to the nearest full time
    df_chartevents["CHARTTIME_FLOOR"] = df_chartevents["CHARTTIME"].dt.floor('1h')
    # Make long df_chartevents to wide data an aggregate by max to keep lables as cols and drop itemids
    df_chartevents_pivot              = df_chartevents.pivot_table(index=["ICUSTAY_ID", "CHARTTIME_FLOOR"], columns=["LABEL"], values="VALUENUM", aggfunc="max")
    df_chartevents_pivot              .reset_index(inplace=True)
    # Merge chart data to created timerange
    df_chartevents_time               = df_time_ranges.merge(df_chartevents_pivot, how="left", on=["ICUSTAY_ID", "CHARTTIME_FLOOR"])
    # Set columnnames to upper letters
    df_chartevents_time.columns       = [i.upper() for i in df_chartevents_time.columns]
    # Save df as csv
    df_chartevents_time               .to_csv(r"./data/temp_data/df_chartevents_time_shaped.csv")
    progress_bar.update(1)
    del df_chartevents_time

    ###########################################################################
    ### 3. Prepare the data from labevents.csv
    ###########################################################################
    # Read labevents.csv
    df_labevents                      = pd.read_csv(r"./data/temp_data/df_labevents_no_outliers.csv", index_col=0)
    # Checking for non-standard date entries in the 'CHARTTIME' column
    # This checks if each date can be converted to datetime, marking as False if it fails
    mask                        = pd.to_datetime(df_labevents['CHARTTIME'], errors='coerce').isna()
    df_labevents                      = df_labevents[~mask]
    # Format charttime to datetime
    df_labevents["CHARTTIME"]         = pd.to_datetime(df_labevents["CHARTTIME"])
    # Create charttime_floor and move charttime to the nearest full time
    df_labevents["CHARTTIME_FLOOR"]   = df_labevents["CHARTTIME"].dt.floor('1h')
    # Make long df_labevents to wide data an aggregate by max to keep lables as cols and drop itemids
    df_labevents_pivot                = df_labevents.pivot_table(index=["HADM_ID", "CHARTTIME_FLOOR"], columns="LABEL", values="VALUENUM", aggfunc="max")
    # Reset index
    df_labevents_pivot.reset_index(inplace=True)
    # Merge lab data to created timerange
    df_labevents_time                 = df_time_ranges.merge(df_labevents_pivot, how="left", on=["HADM_ID", "CHARTTIME_FLOOR"])
    # Set columnnames to upper letters
    df_labevents_time.columns         = [i.upper() for i in df_labevents_time.columns]
    # Save df as csv
    df_labevents_time                 .to_csv(r"./data/temp_data/df_labevents_time_shaped.csv")
    progress_bar.update(1)
    del df_labevents_time

    ###########################################################################
    ### 4. Prepare the data from culture.csv
    ###########################################################################
    # Read culture.csv
    df_culture                          = pd.read_csv(r"./data/extracted_data/culture.csv")
    # Checking for non-standard date entries in the 'CHARTTIME' column
    # This checks if each date can be converted to datetime, marking as False if it fails
    mask                                = pd.to_datetime(df_culture['CULTURE_TIME'], errors='coerce').isna()
    df_culture                          = df_culture[~mask]
    # Format charttime to datetime
    df_culture["CULTURE_TIME"]          = pd.to_datetime(df_culture["CULTURE_TIME"])
    # Create charttime_floor and move charttime to the nearest full time
    df_culture["CULTURE_TIME_FLOOR"]    = df_culture["CULTURE_TIME"].dt.floor('1h')
    # Merge culture data on created timerange to get the ICUSTAY IDs
    df_culture_shaped                   = df_culture.merge(df_time_ranges, how="inner", left_on=["CULTURE_TIME_FLOOR", "HADM_ID"], 
                                                                    right_on=["CHARTTIME_FLOOR", "HADM_ID"])
    df_culture_shaped                   = df_culture_shaped[["ICUSTAY_ID", "CULTURE_TIME", "POSITIVE_CULTURE", "SPEC_ITEMID"]]
    # Set columnnames to upper letters
    df_culture_shaped.columns           = [i.upper() for i in df_culture_shaped.columns]
    # Save df as csv
    df_culture_shaped                   .to_csv(r"./data/temp_data/df_culture_shaped.csv")
    progress_bar.update(1)
    del df_culture_shaped

    ###########################################################################
    ### 5. Prepare the data from output.csv
    ###########################################################################
    # Read output.csv
    df_output                       = pd.read_csv(r"./data/temp_data/df_output_no_outliers.csv", index_col=0)
    # Checking for non-standard date entries in the 'CHARTTIME' column
    # This checks if each date can be converted to datetime, marking as False if it fails
    mask                            = pd.to_datetime(df_output['CHARTTIME'], errors='coerce').isna()
    df_output                       = df_output[~mask]
    # Format charttime to datetime
    df_output["CHARTTIME"]          = pd.to_datetime(df_output["CHARTTIME"])
    # Create charttime_floor and move charttime to the nearest full time
    df_output["CHARTTIME_FLOOR"]    = df_output["CHARTTIME"].dt.floor('1h')
    # Make long df_labevents to wide data an aggregate by max to keep lables as cols and drop itemids
    df_output_pivot                 = df_output.pivot_table(index=["ICUSTAY_ID", "CHARTTIME_FLOOR"], values="VALUENUM", aggfunc="max")
    # Reset index
    df_output_pivot.reset_index(inplace=True)
    # Merge lab data to created timerange
    df_output_time                  = df_time_ranges.merge(df_output_pivot, how="left", on=["ICUSTAY_ID", "CHARTTIME_FLOOR"])
    progress_bar.update(1)

    ###########################################################################
    ### 6. Calculate 24h urine output
    ###########################################################################
    # Function to apply on a pandas group for calc 24h urine output
    def urine_output_24h(group):
        # Sorting values asscending by CHARTTIME FLOOR
        group.sort_values(by='CHARTTIME_FLOOR', inplace=True)
        # If a group has 24h or more with observations the calculation will be applied
        if (group['CHARTTIME_FLOOR'].max() - group['CHARTTIME_FLOOR'].min()).total_seconds()/60/60 >= 24:
            # Create a 24h rolling window
            rolling = group.rolling('24H', on='CHARTTIME_FLOOR')
            # Sum the values of VALUENUM from the rolling window
            group["URINE_OUPUT_24h"] = rolling["VALUENUM"].sum()
            # Setting the first 24h to nan
            group["URINE_OUPUT_24h"] = np.where((group['CHARTTIME_FLOOR'] - group['CHARTTIME_FLOOR'].min()).dt.total_seconds()/60/60 >= 24, group["URINE_OUPUT_24h"], np.nan)
            return group
        # Else the group has less than 24h observations the group is passed with nan for URINE OUTPUT 24h
        else:
            group["URINE_OUPUT_24h"] = np.nan
            return group 
    # Apply the function urine_output_24h on the group
    df_output_time          = df_output_time.groupby(["ICUSTAY_ID"]).apply(urine_output_24h).reset_index(drop=True)
    # Rename Valuenum to urine_ouput_hourly    
    df_output_time.rename(columns={"VALUENUM": "URINE_OUTPUT_HOURLY"}, inplace=True)
    # Save df as csv
    df_output_time          .to_csv(r"./data/temp_data/df_output_time_shaped.csv")
    progress_bar.update(1)
    del df_output_time

    ###########################################################################
    ### 7. Prepare the data from input.csv
    ###########################################################################
    # Read labevents.csv
    df_input                     = pd.read_csv(r"./data/extracted_data/input.csv", index_col=0)
    # Checking for non-standard date entries in the 'CHARTTIME' column
    # This checks if each date can be converted to datetime, marking as False if it fails
    mask                         = pd.to_datetime(df_input['CHARTTIME'], errors='coerce').isna()
    df_input                     = df_input[~mask]
    # Format charttime to datetime
    df_input["CHARTTIME"]        = pd.to_datetime(df_input["CHARTTIME"])
    # Create charttime_floor and move charttime to the nearest full time
    df_input["CHARTTIME_FLOOR"]  = df_input["CHARTTIME"].dt.floor('1h')
    # Make long df_labevents to wide data an aggregate by max to keep lables as cols and drop itemids
    df_input_pivot               = df_input.pivot_table(index=["ICUSTAY_ID", "CHARTTIME_FLOOR"], values="VALUENUM", columns="LABEL", aggfunc="max")
    # Reset index
    df_input_pivot.reset_index(inplace=True)
    # Merge lab data to created timerange
    df_input_time                = df_time_ranges.merge(df_input_pivot, how="left", on=["ICUSTAY_ID", "CHARTTIME_FLOOR"])
    # Set columnnames to upper letters
    df_input_time.columns        = [i.upper() for i in df_input_time.columns]
    # Save df as csv
    df_input_time                .to_csv(r"./data/temp_data/df_input_time_shaped.csv")
    progress_bar.update(1)

    ###########################################################################
    ### 8. Create suspected infection
    ###########################################################################
    # Reading data from .csv
    df_culture_shaped   = pd.read_csv(r"./data/temp_data/df_culture_shaped.csv", index_col=0)
    df_anti_input       = pd.read_csv(r"./data/extracted_data/anti_input.csv", index_col=0)
    df_anti_pres        = pd.read_csv(r"./data/extracted_data/anti_pres.csv", index_col=0)
    # Set the following columns to datetime
    df_culture_shaped["CULTURE_TIME"]       = pd.to_datetime(df_culture_shaped["CULTURE_TIME"])
    df_anti_pres["ANTIBIOTIC_STARTTIME"]    = pd.to_datetime(df_anti_pres["ANTIBIOTIC_STARTTIME"])
    df_anti_pres["ANTIBIOTIC_ENDTIME"]      = pd.to_datetime(df_anti_pres["ANTIBIOTIC_ENDTIME"])
    df_anti_input["ANTIBIOTIC_STARTTIME"]   = pd.to_datetime(df_anti_input["ANTIBIOTIC_STARTTIME"])
    df_anti_input["ANTIBIOTIC_ENDTIME"]     = pd.to_datetime(df_anti_input["ANTIBIOTIC_ENDTIME"])
    # Start by merging ICUSTAY_ID from df_subjects with the minimum CULTURE_TIME from df_culture_shaped.
    # This finds the earliest blood culture time (CULTURE_TIME) for each ICU stay.
    micro       = df_subjects[["ICUSTAY_ID"]].merge(df_culture_shaped.groupby(["ICUSTAY_ID"])["CULTURE_TIME"].min().reset_index(), how="left")
    # Merge ICUSTAY_ID from df_subjects with aggregated minimum and maximum antibiotic_starttime from df_anti_pres.
    # This retrieves the first and last antibiotic prescription times for each ICU stay.
    anti_pres   = df_subjects[["ICUSTAY_ID"]].merge(df_anti_pres.groupby("ICUSTAY_ID").aggregate({"ANTIBIOTIC_STARTTIME":["min", "max"]}).droplevel(axis=1, level=0).reset_index(), how="left")
    anti_input  = df_subjects[["ICUSTAY_ID"]].merge(df_anti_input.groupby("ICUSTAY_ID").aggregate({"ANTIBIOTIC_STARTTIME":["min", "max"]}).droplevel(axis=1, level=0).reset_index(), how="left")
    # Similarly, merge ICUSTAY_ID from df_subjects with aggregated minimum and maximum antibiotic_starttime from df_anti_input.
    # This retrieves the first and last antibiotic input times for each ICU stay, similar to df_anti_pres but for different data.
    df_susp     = anti_input.combine_first(anti_pres).merge(micro)
    # Renaming the columns
    df_susp     .rename(columns={"min":"ANTIBIOTIC_STARTTIME", "max":"ANTIBIOTIC_ENDTIME"}, inplace=True)
    # Calulate the time of suspicion infection
    df_susp["SUSP"] = np.where(
        (df_susp["CULTURE_TIME"].notna()) & (df_susp["ANTIBIOTIC_STARTTIME"].notna()),
        np.where(
            (df_susp["ANTIBIOTIC_STARTTIME"] <= df_susp["CULTURE_TIME"]),
            np.where(
                ((df_susp["CULTURE_TIME"] - df_susp["ANTIBIOTIC_STARTTIME"]) <= pd.Timedelta(hours=24)),
                1,  # Antibiotics started first and BC taken within 24 hours of antibiotics start
                0),
            np.where(
                ((df_susp["ANTIBIOTIC_STARTTIME"] - df_susp["CULTURE_TIME"]) <= pd.Timedelta(hours=72)),
                1,  # BC taken first and antibiotics started within 72 hours of BC
                0)
        ),
        0)  # Either CULTURE_TIME or antibiotic_starttime is missing
    # Set tSUSP to the minimum from bloodculture or antibiotic starttime else not a time
    df_susp['tSUSP']    = pd.to_datetime(np.where(df_susp['SUSP'] == 1, df_susp[['CULTURE_TIME', 'ANTIBIOTIC_STARTTIME']].min(axis=1), pd.NaT))
    # Save df as csv
    df_susp             .to_csv(r"./data/temp_data/df_susp_shaped.csv")
    progress_bar.update(1)
    del df_susp

    ###########################################################################
    ### 9. Prepare the data from vent.csv
    ###########################################################################
    # Reading data from vent.csv
    df_vent                     = pd.read_csv(r"./data/extracted_data/vent.csv", index_col=0) # Read vent.csv
    # Format charttime to datetime
    df_vent["CHARTTIME"]        = pd.to_datetime(df_vent["CHARTTIME"])
    # Create charttime_floor and move charttime to the nearest full time
    df_vent["CHARTTIME_FLOOR"]  = df_vent["CHARTTIME"].dt.floor('1h')
    df_vent                     = df_vent.groupby(["ICUSTAY_ID", "CHARTTIME_FLOOR"]).max().reset_index()
    # Merge vent data to the time template
    df_vent_time                = df_time_ranges.merge(df_vent[["ICUSTAY_ID", "CHARTTIME_FLOOR", "MECHVENT"]], how="left", on=["ICUSTAY_ID", "CHARTTIME_FLOOR"])
    # Set columnnames to upper letters
    df_vent_time.columns        = [i.upper() for i in df_vent_time.columns]
    # Save df as csv
    df_vent_time                .to_csv(r"./data/temp_data/df_vent_time_shaped.csv")
    progress_bar.update(1)
    del df_vent_time


def main():
    data_preprocessing()
    
if __name__ == '__main__':
    main()

