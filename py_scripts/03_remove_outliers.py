import pandas as pd
import warnings
warnings.simplefilter('ignore')

def remove_outliers():
    ###########################################################################
    ### Function for removing outliers by IQR
    ###########################################################################
    def remove_outliers_iqr(group, skip_group_name=None):
        """
        Function for removing outliers
        from extracted mimic-iii data.\n
        Parameters:
            group: e.g. df_chartevents.groupby("LABEL").apply(lambda group: remove_outliers_iqr(group, skip_group_name=['GCS', 'FiO2']))
            skip_group_name: list of Lables to skip e.g. ['GCS', 'FiO2']
        Returns:
            Pandas Dataframe
        """
        if skip_group_name is not None and any(name_part in group.name for name_part in skip_group_name):
            return group
        else:
            Q1  = group['VALUENUM'].quantile(0.25)
            Q3  = group['VALUENUM'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR 
            upper_bound = Q3 + 1.5 * IQR 
            return group[(group['VALUENUM'] >= lower_bound) & (group['VALUENUM'] <= upper_bound)]
    
    ###########################################################################
    ### 1. Remove outliers
    ###########################################################################
    # Read chartevents.csv
    df_chartevents  = pd.read_csv(r"./data/extracted_data/chartevents.csv", index_col=0)
    # Remove Outliers
    df_chartevents  = df_chartevents.groupby("LABEL").apply(lambda group: remove_outliers_iqr(group, skip_group_name=['GCS', 'FiO2'])).reset_index(drop=True) # Skip GCS and FIO2
    df_chartevents  = df_chartevents[df_chartevents["VALUENUM"] >= 0] # All Chartevents must be larger or equal than 0
    df_chartevents.loc[(df_chartevents["ITEMID"] == 190) & (df_chartevents["VALUENUM"] < 21), "VALUENUM"] = 21 # FIO2 if lower than 21 set to 21 for ITEMID 190

    df_chartevents  .to_csv(r"./data/temp_data/df_chartevents_no_outliers.csv")
    del df_chartevents
   
    # Read labevents.csv
    df_labevents    = pd.read_csv(r"./data/extracted_data/labevents.csv", index_col=0)
    # Remove Outliers
    df_labevents    = df_labevents.groupby("LABEL").apply(lambda group: remove_outliers_iqr(group)).reset_index(drop=True)
    df_labevents    = df_labevents[~((df_labevents["LABEL"] == "so2_bloodgas") & (df_labevents["VALUENUM"] > 100))] # o2 saturation can not be greater than 100
    df_labevents    = df_labevents[df_labevents["VALUENUM"] >= 0] # Labevents cant be negative
    df_labevents    .to_csv(r"./data/temp_data/df_labevents_no_outliers.csv")
    del df_labevents
    
    # Read output.csv
    df_output   = pd.read_csv(r"./data/extracted_data/output.csv", index_col=0)
    # Remove Outliers
    df_output   = df_output.groupby("ITEMID").apply(lambda group: remove_outliers_iqr(group)).reset_index(drop=True)
    df_output   = df_output[df_output["VALUENUM"] >= 0]
    df_output   .to_csv(r"./data/temp_data/df_output_no_outliers.csv")
    del df_output

def main():
    remove_outliers()
    
if __name__ == '__main__':
    main()

