import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
warnings.simplefilter('ignore')

def handle_missing_data():
    ###########################################################################
    ### 1. Fill chart data
    ###########################################################################
    # Missing Chartevents are handled with linear interpolation, only FIO2 and
    # GCS are imputed with ffill and bfill
    def transform_chunk(group): # This is the function for handle the missing data
        non_gcs_cols    = [col for col in group.columns if "GCS" not in col and "FIO2" not in col] # Hold non GCS and FIO2 cols
        gcs_fio2_cols   = [col for col in group.columns if "GCS" in col or "FIO2" in col] # Hold only GCS and FIO2 cols
        
        group_non_gcs = group.groupby("ICUSTAY_ID")[non_gcs_cols].apply(
            lambda group: group.interpolate(method="linear", limit_direction="both")
        ).reset_index(drop=True) # The lineare interploation is applied to non_gcs_cols
        
        group_gcs_fio2 = group.groupby("ICUSTAY_ID")[gcs_fio2_cols].apply(
            lambda group: group.ffill().bfill()
        ).reset_index(drop=True) # The ffill and bfill imputation is applied to gcs_fio2_cols
        
        group_final = pd.concat([group_non_gcs, group_gcs_fio2], axis=1) # Putting the dataframe back togehter
        return group_final

    def fill_df(df): # Function to apply the imputaion transform_chunk on the ICUSTAY and show the in a progress bar
        chunks = [group for _, group in df.groupby('ICUSTAY_ID')]
        transformed_chunks = [transform_chunk(chunk) for chunk in tqdm(chunks, total=len(chunks), desc="Progress")]
        return pd.concat(transformed_chunks).reset_index(drop=True)

    df          = pd.read_csv(r"./data/temp_data/df_chartevents_time_reduced.csv", index_col=0) # Reading the data
    df_filled   = fill_df(df) # Apply the imputation
    df_filled   .to_csv(r"./data/temp_data/df_chartevents_time_filled.csv") # Save Imputed df
    del df
    del df_filled

    ###########################################################################
    ### 2. Backward and forward fill lab data
    ###########################################################################
    # Missing Labevents are handled with ffill and bfill
    def transform_chunk(group):
        group = group.groupby("ICUSTAY_ID").apply(lambda group: group.ffill().bfill()).reset_index(drop=True)
        return group

    def fill_df(df):
        chunks = [group for _, group in df.groupby('ICUSTAY_ID')]
        transformed_chunks = [transform_chunk(chunk) for chunk in tqdm(chunks, total=len(chunks), desc="Progress")]
        return pd.concat(transformed_chunks).reset_index(drop=True)

    df          = pd.read_csv(r"./data/temp_data/df_labevents_time_reduced.csv", index_col=0) # Reading the data
    df_filled   = fill_df(df) # Apply the imputation
    df_filled   .to_csv(r"./data/temp_data/df_labevents_time_filled.csv") # Save Imputed df
    del df
    del df_filled

    ###########################################################################
    ### 3. Intterative Imputation < 20% missing Values
    ###########################################################################
    # Class for avoid negative Imputation
    class NonNegativeImputer(TransformerMixin, BaseEstimator):
        def __init__(self, imputer):
            self.imputer = imputer

        def fit(self, X, y=None):
            self.imputer.fit(X, y)
            return self

        def transform(self, X):
            X_imputed = self.imputer.transform(X)
            X_imputed[X_imputed < 0] = 0
            return X_imputed

    df_chartevents  = pd.read_csv(r"./data/temp_data/df_chartevents_time_filled.csv", index_col=0) # Reading data
    df_labevents    = pd.read_csv(r"./data/temp_data/df_labevents_time_filled.csv", index_col=0) # Reading data
    df_impute       = df_chartevents.merge(df_labevents, on=["CHARTTIME_FLOOR", "HADM_ID", "ICUSTAY_ID", "HOUR"], how="left") # Merging both df for imputation

    # Mask cols with less 20% of missing data
    lower_mask  = (df_impute.iloc[:,4:].isna().mean() < 0.2) 

    # NonNegativeImputer applied
    imputer = NonNegativeImputer(IterativeImputer(HistGradientBoostingRegressor(random_state=42), 
                                                  max_iter=10, random_state=42, verbose=3))
    df_impute.loc[:,lower_mask[lower_mask == True].index] = imputer.fit_transform(df_impute.loc[:,lower_mask[lower_mask == True].index])

    # Save imputed data
    df_chartevents_imputed  = df_impute.loc[:,df_chartevents.columns].dropna(axis=1) # Save imputed data as chartevents by locating og cols
    df_chartevents_imputed  .to_csv(r"./data/temp_data/df_chartevents_time_imputed.csv")
    df_labevents_imputed    = df_impute.loc[:,df_labevents.columns].dropna(axis=1) # Save imputed data as labevents by locating og cols
    df_labevents_imputed    .to_csv(r"./data/temp_data/df_labevents_time_imputed.csv")
    del df_impute

    ###########################################################################
    ### 4. Handle missing data in vent
    ###########################################################################
    df_vent_time_reduced            = pd.read_csv(r"./data/temp_data/df_vent_time_reduced.csv", index_col=0) # Reading data
    df_vent_time_imputed            = df_vent_time_reduced.groupby("ICUSTAY_ID").apply(lambda group: group.ffill()).reset_index(drop=True) # Impute missing Vent data with ffill
    df_vent_time_imputed["MECHVENT"].fillna(0, inplace=True) # Impute rest of missing Vent data with 0
    df_vent_time_imputed            .to_csv(r"./data/temp_data/df_vent_time_imputed.csv") # Save imputed data
    del df_vent_time_imputed

    ##########################################################################
    ## 5. Handle missing data in input
    ##########################################################################
    df_input_time_reduced   = pd.read_csv(r"./data/temp_data/df_input_time_reduced.csv", index_col=0) # Reading data
    for col in df_input_time_reduced.columns:
        df_input_time_reduced[col].fillna(0, inplace=True) # Impute missing data with 0
    df_input_time_imputed   = df_input_time_reduced.copy()
    df_input_time_imputed.to_csv(r"./data/temp_data/df_input_time_imputed.csv") # Save imputed data
    del df_input_time_imputed

    ##########################################################################
    ## 6. Handle missing data in output
    ##########################################################################
    df_output_time_reduced  = pd.read_csv(r"./data/temp_data/df_output_time_reduced.csv", index_col=0) # Reading data
    for col in df_output_time_reduced.columns:
        df_output_time_reduced[col].fillna(0, inplace=True) # Impute missing data with 0
    df_output_time_imputed  = df_output_time_reduced.copy()
    df_output_time_imputed  .to_csv(r"./data/temp_data/df_output_time_imputed.csv") # Save imputed data
    del df_output_time_imputed


def main():
    handle_missing_data()
    
if __name__ == '__main__':
    main()

