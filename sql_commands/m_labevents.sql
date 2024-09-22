-- 

DROP TABLE IF EXISTS m_labevents; 

CREATE TABLE m_labevents AS
WITH labeled_data AS (
    SELECT
        subject_id,
        hadm_id,
        charttime,
        itemid,
        CASE
            WHEN itemid = 50868 THEN 'ANION_GAP' -- AND valuenum <= 10000
            WHEN itemid = 50862 THEN 'ALBUMIN' -- AND valuenum <= 10
            WHEN itemid = 51144 THEN 'BANDS' -- AND valuenum BETWEEN 0 AND 100
            WHEN itemid = 50882 THEN 'BICARBONATE' -- AND valuenum <= 10000
            WHEN itemid = 50885 THEN 'BILIRUBIN' -- AND valuenum <= 150
            WHEN itemid = 50912 THEN 'CREATININE' -- AND valuenum <= 150
            WHEN itemid IN (50806, 50902) THEN 'CHLORIDE' -- AND valuenum <= 10000
            -- WHEN itemid IN (50809, 50931) THEN 'GLUCOSE' -- AND valuenum <= 10000 comes from charteventdata
            WHEN itemid IN (50810, 51221) THEN 'HEMATOCRIT' -- AND valuenum <= 100
            WHEN itemid IN (50811, 51222) THEN 'HEMOGLOBIN' -- AND valuenum <= 50
            WHEN itemid = 50813 THEN 'LACTATE' -- AND valuenum <= 50
            WHEN itemid = 51265 THEN 'PLATELET' -- AND valuenum <= 10000
            WHEN itemid IN (50822, 50971) THEN 'POTASSIUM' -- AND valuenum <= 30
            WHEN itemid = 51275 THEN 'PTT' -- AND valuenum <= 150
            WHEN itemid = 51237 THEN 'INR' -- AND valuenum <= 50
            WHEN itemid = 51274 THEN 'PT' -- AND valuenum <= 150
            WHEN itemid IN (50824, 50983) THEN 'SODIUM' -- AND valuenum <= 200
            WHEN itemid = 51006 THEN 'BUN' -- AND valuenum <= 300
            WHEN itemid IN (51300, 51301) THEN 'WBC' -- AND valuenum <= 1000
            WHEN itemid = 50924 THEN 'FERRITIN' -- AND valuenum <= 10000
            WHEN itemid = 50998 THEN 'TRANSFERRIN' -- AND valuenum <= 10000
            WHEN itemid = 50910 THEN 'CREATINE_KINASE' -- AND valuenum <= 20000
            WHEN itemid = 50911 THEN 'CK_MB' -- AND valuenum <= 10000
            WHEN itemid = 50915 THEN 'D_DIMER' -- AND valuenum <= 100000
            WHEN itemid = 50963 THEN 'NTPROBNP' -- AND valuenum <= 100000
            WHEN itemid = 51288 THEN 'SEDIMENTATION_RATE' -- AND valuenum <= 5000
            WHEN itemid = 51214 THEN 'FIBRINOGEN' -- AND valuenum <= 10000
            WHEN itemid = 50954 THEN 'LDH' -- AND valuenum <= 10000
            WHEN itemid = 50960 THEN 'MAGNESIUM' -- AND valuenum <= 50
            WHEN itemid = 50808 THEN 'CALCIUM_FREE' -- AND valuenum <= 50
            WHEN itemid = 50821 THEN 'PO2_BLOODGAS' -- AND valuenum <= 1000
            WHEN itemid = 50820 THEN 'PH_BLOODGAS' -- AND valuenum <= 15
            WHEN itemid = 50818 THEN 'PCO2_BLOODGAS' -- AND valuenum <= 500
            WHEN itemid = 50817 THEN 'SO2_BLOODGAS' -- AND valuenum <= 100
            WHEN itemid = 51003 THEN 'TROPONIN_T' -- AND valuenum <= 100
            WHEN itemid = 51014 THEN 'GLUCOSE_CSF' -- AND valuenum <= 10000
            WHEN itemid = 51024 THEN 'TOTAL_PROTEIN_JOINT_FLUID' -- AND valuenum <= 100
            WHEN itemid = 51059 THEN 'TOTAL_PROTEIN_PLEURAL' -- AND valuenum <= 100
            WHEN itemid = 51070 THEN 'URINE_ALBUMIN_CREATININE_RATIO' -- AND valuenum <= 100000
            WHEN itemid = 51128 THEN 'WBC_ASCITES' -- AND valuenum <= 100000
            WHEN itemid = 50889 THEN 'CRP' -- AND valuenum <= 366
            -- WHEN itemid = 50816 THEN 'FIO2' -- AND valuenum <= 100
            -- WHEN itemid = 50815 THEN 'O2FLOW' -- AND valuenum <= 70


        END AS label,
        valuenum
    FROM
        labevents
    WHERE
        valuenum IS NOT NULL
    AND itemid IN (50868, 50862, 51144, 50882, 50885, 
                    50912, 50806, 50902, 50809, 50931, 
                    50810, 51221, 50811, 51222, 50813, 
                    51265, 50822, 50971, 51275, 51237, 
                    51274, 50824, 50983, 51006, 51300, 
                    51301, 50924, 50998, 50910, 50911, 
                    50915, 50963, 51288, 51214, 50954, 
                    50960, 50808, 50821, 50820, 50818, 
                    50817, 51003, 51014, 51024, 51059, 
                    51070, 51128, 50889)
)

SELECT
    subject_id,
    hadm_id,
    charttime,
    itemid,
    label,
    valuenum
FROM
    labeled_data ld
WHERE
    label IS NOT NULL
AND hadm_id IS NOT NULL
AND charttime IS NOT NULL
AND valuenum IS NOT NULL
ORDER BY
   subject_id, hadm_id, charttime;
