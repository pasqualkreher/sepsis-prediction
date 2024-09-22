DROP TABLE IF EXISTS m_subjects; 
CREATE TABLE m_subjects AS

WITH co_dx AS
(
    SELECT 
        hadm_id,
        MAX(
            CASE
                WHEN icd9_code = '99591' THEN 1   -- Sepsis not specified
                ELSE 0 
            END
        ) AS unspecified_sepsis,
        MAX(
            CASE
                WHEN icd9_code = '99592' THEN 1   -- Severe sepsis
                ELSE 0 
            END
        ) AS severe_sepsis,
        MAX(
            CASE
                WHEN icd9_code = '78552' THEN 1   -- Septic shock
                ELSE 0 
            END
        ) AS septic_shock
    FROM diagnoses_icd
    GROUP BY hadm_id
),

stay AS
(
    SELECT
        subject_id,
        hadm_id,
        icustay_id,
        dbsource,
        intime,
        outtime,
        first_careunit,
        last_careunit                  
    FROM icustays                    
),

pat AS
(
    SELECT
        subject_id,
        gender,
        dob,
        dod,
        dod_hosp
    FROM patients
)

, t1 AS 
(
    SELECT
        adm.subject_id,
        adm.hadm_id,
        stay.icustay_id,
        adm.diagnosis,
        stay.intime,
        stay.outtime,
        adm.admittime,
        adm.dischtime,
        adm.ethnicity,
        pat.gender,
        pat.dob,
        pat.dod,
        pat.dod_hosp,
        CASE 
            WHEN round((cast(adm.admittime as date) - cast(pat.dob as date)) / 365.242, 4) > 89 
                THEN 91.4
            ELSE round((cast(adm.admittime as date) - cast(pat.dob as date)) / 365.242, 4) 
        END as age,
        stay.first_careunit,
        stay.last_careunit,
        stay.dbsource,
        co_dx.unspecified_sepsis,
        co_dx.severe_sepsis,
        co_dx.septic_shock,
        CASE 
            WHEN co_dx.unspecified_sepsis = 1 OR co_dx.severe_sepsis = 1 OR co_dx.septic_shock = 1
                THEN 1
            ELSE 0 
        END AS sepsis
    FROM admissions adm
    INNER JOIN co_dx
        ON adm.hadm_id = co_dx.hadm_id
    INNER JOIN stay
        ON adm.hadm_id = stay.hadm_id
    INNER JOIN pat
        ON adm.subject_id = pat.subject_id
    ORDER BY adm.subject_id, adm.hadm_id
)

SELECT *
FROM t1
WHERE age > 1
;
