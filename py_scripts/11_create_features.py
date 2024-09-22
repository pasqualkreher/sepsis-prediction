import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.simplefilter('ignore')

def make_lag_stats(df, chartevents_cols, output_cols, calc_cols, lags, 
                    stats=['mean', 'std', 'var', 'min', 'max']): 
    """
    Create lag statistics features for each lag and statistic \n
    from the chart data.
    """
    og_cols = df.columns
    cols = list(chartevents_cols) + list(output_cols) + list(calc_cols)
    selected_cols = []
    for col in cols:
        if col in df.columns:
            selected_cols.append(col)    

    # Erstellt für jeden Statistiktyp und jedes Lag eine Liste von DataFrames
    results = []

    for stat in stats:
        for lag in lags:
            # Berechne die Statistik für alle Spalten und speichere sie in einem DataFrame
            grouped         = df.groupby("ICUSTAY_ID")[selected_cols].rolling(window=lag).agg(stat).reset_index(level=0, drop=True)
            grouped.columns = [f"{stat}_l{lag-1}_" + col for col in grouped.columns]
            expanding_stat  = df.groupby("ICUSTAY_ID")[selected_cols].expanding().agg(stat).reset_index(level=0, drop=True)            
            for grouped_col in grouped.columns:
                is_na = grouped[grouped_col].isna()
                grouped.loc[is_na, grouped_col] = expanding_stat.loc[is_na, grouped_col.replace(f"{stat}_l{lag-1}_", "")]
            grouped.fillna(0, inplace=True)
            results.append(grouped)

    results.insert(0, df)
    df_final = pd.concat(results, axis=1)
    df_final.drop(columns=og_cols, inplace=True)
    return df_final

def make_first_features(df, labevents_cols):
    """
    Create features for each chart and lab data, \n
    which hold first values and the diff from the current \n
    value to the first value.
    """
    og_cols         = df.columns
    cols            = list(labevents_cols)
    selected_cols   = [col for col in cols if col in df.columns]

    df_first = df.groupby("ICUSTAY_ID", as_index=False)[selected_cols].first()
    df_first.set_index('ICUSTAY_ID', inplace=True)
    df_first.columns    = ["first_" + colname for colname in df_first.columns]
    df_first_merged     = df.join(df_first, on="ICUSTAY_ID")
    df_delta_to_first   = pd.DataFrame((df[selected_cols].values - df_first_merged.loc[:, ["first_" + col for col in selected_cols]].values),
                                        columns=["delta_" + colname for colname in selected_cols],
                                        index=df.index)
    df_final = pd.concat([df_first_merged, df_delta_to_first], axis=1)
    df_final.drop(columns=og_cols, inplace=True) 
    return df_final

def make_lags(df, chartevents_cols, output_cols, calc_cols, lags): 
    """
    Create shifted features from chart data by the given lags.
    """
    og_cols         = df.columns
    cols            = list(chartevents_cols) + list(output_cols) + list(calc_cols)
    selected_cols   = [col for col in cols if col in df.columns]

    results = []

    for lag in lags:
        grouped = df.groupby("ICUSTAY_ID")[selected_cols].apply(lambda group: group.shift(lag))
        grouped.bfill(inplace=True)
        grouped.columns = [f"shift_{lag}_" + col for col in grouped.columns]
        grouped.index = grouped.reset_index()["level_1"]
        grouped.fillna(0, inplace=True)
        results.append(grouped)
    results.insert(0, df)
    df_final = pd.concat(results, axis=1)
    df_final.drop(columns=og_cols, inplace=True)
    return df_final

def make_lag_diff(df, chartevents_cols, output_cols, calc_cols,  lags):
    """
    Create features form chart data which are shifted \n
    by the given lags and the diff is caluated between \n
    the current value and the shifted value.
    """
    og_cols         = df.columns
    cols            = list(chartevents_cols) + list(output_cols) + list(calc_cols)
    selected_cols   = [col for col in cols if col in df.columns]

    results = []

    for lag in lags:
        grouped = df.groupby("ICUSTAY_ID")[selected_cols].apply(
            lambda group: group - group.shift(lag)
        )
        grouped.fillna(0, inplace=True)
        grouped.columns = [f"diff_{lag}_" + col for col in grouped.columns]
        grouped.index = grouped.reset_index()["level_1"]
        results.append(grouped)

    results.insert(0, df)
    df_final = pd.concat(results, axis=1)
    df_final.drop(columns=og_cols, inplace=True)
    return df_final

def make_expo(df, chartevents_cols, calc_cols, exponents):
    og_cols         = df.columns
    cols            = list(chartevents_cols) + list(calc_cols)
    selected_cols   = [col for col in cols if col in df.columns]

    results = []

    for expo in exponents:
        df_selected = df[selected_cols]**expo
        df_selected.columns = [f"expo{expo}_" + col for col in df_selected.columns]
        results.append(df_selected)
    df_selected = pd.concat(results, axis=1)
    df_final    = pd.concat([df, df_selected], axis=1)
    df_final    .drop(columns=og_cols, inplace=True)
    return df_final

def adm_cluster_diagnosis():
    df_subjects     = pd.read_csv(r"./data/target_data/df_subjects.csv", index_col=0)
    df_sepsis_flag  = pd.read_csv(r"./data/target_data/df_sepsis_flag.csv", index_col=0)
    df              = df_subjects.merge(df_sepsis_flag.groupby("ICUSTAY_ID")[["SEPSIS_FLAG"]].max().reset_index())

    diagnosis               = df.groupby(["DIAGNOSIS"])[["ICUSTAY_ID"]].count().sort_values(by="ICUSTAY_ID", ascending=False).reset_index()
    vectorizer              = TfidfVectorizer(stop_words='english')
    X                       = vectorizer.fit_transform(diagnosis['DIAGNOSIS'])
    kmeans                  = KMeans(n_clusters=50, random_state=42)
    diagnosis['CLUSTER']    = kmeans.fit_predict(X)

    diagnosis           = diagnosis[['DIAGNOSIS', 'CLUSTER']]
    df                  = df.merge(diagnosis, on='DIAGNOSIS', how='left')
    df["CLUSTER"]       .fillna(9999, inplace=True)
    diagnosis_dummies   = pd.get_dummies(df["CLUSTER"], dtype="int") 
    df                  = pd.concat([df, diagnosis_dummies], axis=1)
    cols                = diagnosis_dummies.columns.to_list()
    cols                .insert(0, "ICUSTAY_ID")
    df                  = df[cols]

    cluster_names = {
        0: 'Coronary_Artery_Diseases',
        1: 'Diabetes_and_Complications',
        2: 'Chronic_Obstructive_Pulmonary_Disease',
        3: 'Asthma_and_Respiratory_Issues',
        4: 'Heart_Failure_and_Related_Conditions',
        5: 'Renal_Failure_and_Kidney_Issues',
        6: 'Stroke_and_Cerebrovascular_Diseases',
        7: 'Hypertension',
        8: 'Gastrointestinal_Disorders',
        9: 'Chest_Pain_and_Cardiac_Symptoms',
        10: 'Arrhythmias',
        11: 'Cancer',
        12: 'Infections',
        13: 'Surgical_Procedures',
        14: 'Liver_Diseases',
        15: 'Endocrine_Disorders',
        16: 'Respiratory_Infections',
        17: 'Urinary_Tract_Infections',
        18: 'Mental_Health_Disorders',
        19: 'Anemia_and_Blood_Disorders',
        20: 'Musculoskeletal_Disorders',
        21: 'Obesity',
        22: 'Autoimmune_Diseases',
        23: 'Dermatological_Conditions',
        24: 'Neurological_Disorders',
        25: 'Osteoporosis',
        26: 'Ophthalmologic_Conditions',
        27: 'Pediatric_Conditions',
        28: 'Gynecological_Disorders',
        29: 'Pregnancy_and_Childbirth',
        30: 'Hematologic_Malignancies',
        31: 'Allergies',
        32: 'Rheumatic_Diseases',
        33: 'Chronic_Pain_Conditions',
        34: 'Cardiomyopathy',
        35: 'Vascular_Diseases',
        36: 'Alcohol_and_Substance_Abuse',
        37: 'Pancreatic_Disorders',
        38: 'Lipid_Metabolism_Disorders',
        39: 'Bone_Fractures',
        40: 'Psychosomatic_Disorders',
        41: 'Immune_Deficiencies',
        42: 'Gastroesophageal_Reflux_Disease_(GERD)',
        43: 'Sleep_Disorders',
        44: 'Injury_and_Trauma',
        45: 'Inborn_Errors_of_Metabolism',
        46: 'Infectious_Diseases_(General)',
        47: 'Peripheral_Neuropathy',
        48: 'Burns_and_Skin_Injuries',
        49: 'Emergency_Conditions'
    }

    col_names   = df.columns[1:-1].map(cluster_names).to_list()
    col_names   .insert(0,"ICUSTAY_ID")
    col_names   .extend(["OTHER"])
    col_names   = [i.upper() for i in col_names]
    df.columns  = col_names
    return df

def calculate_features(df):
    df["GCSTOTAL"]                       = df["GCSEYES"] + df["GCSMOTOR"] + df["GCSVERBAL"]
    df["PO2_BLOODGAS_per_FIO2"]          = df["PO2_BLOODGAS"] * 100 / df["FIO2"]
    df["DIABPxMEANBP"]                   = df["DIABP"] * df["MEANBP"]
    df["DIABPxSYSBP"]                    = df["DIABP"] * df["SYSBP"]
    df["HEARTRATExRESPRATE"]             = df["HEARTRATE"] * df["RESPRATE"]
    df["SPO2_PULSOXYxPO2_BLOODGAS"]      = df["SPO2_PULSOXY"] * df["PO2_BLOODGAS"]
    df["SPO2_PULSOXYxMECHVENT"]          = df["SPO2_PULSOXY"] * df["MECHVENT"]
    df["SPO2_PULSOXYxGCSTOTAL"]          = df["SPO2_PULSOXY"] * df["GCSTOTAL"]
    df["ANION_GAPxBICARBONATE"]          = df["ANION_GAP"] * df["BICARBONATE"]
    df["ANION_GAPxBUN"]                  = df["ANION_GAP"] * df["BUN"]
    df["ANION_GAPxCREATININE"]           = df["ANION_GAP"] * df["CREATININE"]
    df["BICARBONATExCHLORIDE"]           = df["BICARBONATE"] * df["CHLORIDE"]
    df["BICARBONATExPCO2_BLOODGAS"]      = df["BICARBONATE"] * df["PCO2_BLOODGAS"]
    df["BICARBONATExPH_BLOODGAS"]        = df["BICARBONATE"] * df["PH_BLOODGAS"]
    df["CHLORIDExSODIUM"]                = df["CHLORIDE"] * df["SODIUM"]
    df["CREATININExPH_BLOODGAS"]         = df["CREATININE"] * df["PH_BLOODGAS"]
    df["CREATININExPOTASSIUM"]           = df["CREATININE"] * df["POTASSIUM"]
    df["CREATININExURINE_OUTPUT_HOURLY"] = df["CREATININE"] * df["URINE_OUTPUT_HOURLY"]
    df["HEMATOCRITxHEMOGLOBIN"]          = df["HEMATOCRIT"] * df["HEMOGLOBIN"]
    df["INRxPT"]                         = df["INR"] * df["PT"]
    df["INRxPTT"]                        = df["INR"] * df["PTT"]
    df["PCO2_BLOODGASxPH_BLOODGAS"]      = df["PCO2_BLOODGAS"] * df["PH_BLOODGAS"]
    df["PLATELETxWBC"]                   = df["PLATELET"] * df["WBC"]
    df["PO2_BLOODGASxGCSTOTAL"]          = df["PO2_BLOODGAS"] * df["GCSTOTAL"]
    df["PTTxPT"]                         = df["PTT"] * df["PT"]
    df["URINE_OUTPUT_HOURLYxANION_GAP"]  = df["URINE_OUTPUT_HOURLY"] * df["ANION_GAP"]
    df["AGExDIABP"]                      = df["AGE"] * df["DIABP"]
    return df


def make_features_ml():
    df_chartevents  = pd.read_csv(r"./data/target_data/df_chartevents_time.csv", index_col=0)
    df_labevents    = pd.read_csv(r"./data/target_data/df_labevents_time.csv", index_col=0)
    df_output       = pd.read_csv(r"./data/target_data/df_output_time.csv", index_col=0)
    df_ml_reduced   = pd.read_csv(r"./data/temp_data/df_ml_reduced.csv", index_col=0)
    df_ml_reduced   = calculate_features(df_ml_reduced)
    
    chartevents_cols    = df_chartevents.columns[4:]
    labevents_cols      = df_labevents.columns[4:]
    output_cols         = df_output.columns[4:]
    calc_cols           = ["GCSTOTAL", "PO2_BLOODGAS_per_FIO2", "DIABPxMEANBP", "DIABPxSYSBP", "HEARTRATExRESPRATE", "SPO2_PULSOXYxPO2_BLOODGAS", 
                           "SPO2_PULSOXYxMECHVENT", "SPO2_PULSOXYxGCSTOTAL", "CREATININExURINE_OUTPUT_HOURLY", 
                           "PO2_BLOODGASxGCSTOTAL", "URINE_OUTPUT_HOURLYxANION_GAP", "AGExDIABP"]
    del df_chartevents
    del df_labevents

    df_lag_stats    = make_lag_stats(df_ml_reduced, chartevents_cols, output_cols, calc_cols, lags=[5,13], stats=['mean', 'std', 'min', 'max'])
    df_lags         = make_lags(df_ml_reduced, chartevents_cols, output_cols, calc_cols, lags=[4,12])
    df_lag_diff     = make_lag_diff(df_ml_reduced, chartevents_cols, output_cols, calc_cols, lags=[4,12])
    df_expo         = make_expo(df_ml_reduced, chartevents_cols, calc_cols, exponents=[2])
    df_first        = make_first_features(df_ml_reduced, labevents_cols)
    df_ml           = pd.concat([df_ml_reduced, df_lag_stats, df_lags, df_lag_diff, df_expo, df_first], axis=1)
    df_ml           = df_ml.merge(adm_cluster_diagnosis())
    df_ml           .replace([np.inf, -np.inf], np.nan, inplace=True)
    df_ml           .fillna(0, inplace=True)
    df_ml           .to_csv(r"./data/target_data/df_ml.csv")

def make_features_ml_testing():
    df_chartevents  = pd.read_csv(r"./data/target_data/df_chartevents_time.csv", index_col=0)
    df_labevents    = pd.read_csv(r"./data/target_data/df_labevents_time.csv", index_col=0)
    df_output       = pd.read_csv(r"./data/target_data/df_output_time.csv", index_col=0)
    df_ml_reduced   = pd.read_csv(r"./data/temp_data/df_ml_testing_reduced.csv", index_col=0)
    df_ml_reduced   = calculate_features(df_ml_reduced)

    chartevents_cols    = df_chartevents.columns[4:]
    labevents_cols      = df_labevents.columns[4:]
    output_cols         = df_output.columns[4:]
    calc_cols           = ["GCSTOTAL", "PO2_BLOODGAS_per_FIO2", "DIABPxMEANBP", "DIABPxSYSBP", "HEARTRATExRESPRATE", "SPO2_PULSOXYxPO2_BLOODGAS", 
                           "SPO2_PULSOXYxMECHVENT", "SPO2_PULSOXYxGCSTOTAL", "CREATININExURINE_OUTPUT_HOURLY", 
                           "PO2_BLOODGASxGCSTOTAL", "URINE_OUTPUT_HOURLYxANION_GAP", "AGExDIABP"]
    del df_chartevents
    del df_labevents

    df_lag_stats    = make_lag_stats(df_ml_reduced, chartevents_cols, output_cols, calc_cols, lags=[5,13], stats=['mean', 'std', 'min', 'max'])
    df_lags         = make_lags(df_ml_reduced, chartevents_cols, output_cols, calc_cols, lags=[4,12])
    df_lag_diff     = make_lag_diff(df_ml_reduced, chartevents_cols, output_cols, calc_cols, lags=[4,12])
    df_expo         = make_expo(df_ml_reduced, chartevents_cols, calc_cols, exponents=[2])
    df_first        = make_first_features(df_ml_reduced, labevents_cols)
    df_ml           = pd.concat([df_ml_reduced, df_lag_stats, df_lags, df_lag_diff, df_expo, df_first], axis=1)
    df_ml           = df_ml.merge(adm_cluster_diagnosis())
    df_ml           .replace([np.inf, -np.inf], np.nan, inplace=True)
    df_ml           .fillna(0, inplace=True)
    df_ml           .to_csv(r"./data/target_data/df_ml_testing.csv")


def main():
    make_features_ml()
    make_features_ml_testing()

if __name__ == '__main__':
    main()