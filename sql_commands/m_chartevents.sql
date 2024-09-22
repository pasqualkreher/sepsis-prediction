DROP TABLE IF EXISTS m_chartevents;

CREATE TABLE m_chartevents AS
WITH labeled_data AS (
    SELECT
        icustay_id,
        charttime,
        itemid,
        CASE
            WHEN itemid IN (211, 220045) THEN 'HeartRate' -- AND valuenum > 0 AND valuenum < 300
            WHEN itemid IN (51, 442, 455, 6701, 220179, 220050, 225309) THEN 'SysBP' -- AND valuenum > 0 AND valuenum < 400 
            WHEN itemid IN (8368, 8440, 8441, 8555, 220180, 220051, 225310) THEN 'DiaBP' -- AND valuenum > 0 AND valuenum < 300
            WHEN itemid IN (220052, 220181, 225312, 52, 6702, 456) THEN 'MeanBP' -- AND valuenum > 0 AND valuenum < 300 
            WHEN itemid IN (615, 618, 220210, 224690) THEN 'RespRate' -- AND valuenum > 0 AND valuenum < 70 
            WHEN itemid IN (223761, 678, 223762, 676) THEN 'TempC'
            WHEN itemid IN (646, 220277) THEN 'SpO2_pulsoxy' -- valuenum > 0 AND valuenum <= 100
            WHEN itemid IN (223835, 3420, 190, 3422) THEN 'FiO2'
            -- WHEN itemid = 227444 THEN 'CRP'  -- CRP comes from lab data
            WHEN itemid IN (807, 811, 1529, 225664, 220621, 226537) THEN 'Glucose' -- AND valuenum > 0 AND valuenum < 2000 
            WHEN itemid IN (723, 223900) THEN 'GCSVerbal'
            WHEN itemid IN (454, 223901) THEN 'GCSMotor'
            WHEN itemid IN (184, 220739) THEN 'GCSEyes'
            ELSE NULL
        END AS label,
        CASE
            -- WHEN itemid IN (223761, 678) AND valuenum > 70 AND valuenum < 120 THEN valuenum  -- TempF
            -- WHEN itemid IN (223762, 676) AND valuenum > 10 AND valuenum < 50 THEN valuenum  -- TempC
            WHEN itemid IN (223761, 678) THEN (valuenum - 32) / 1.8  -- Conversion for Fahrenheit to Celsius
            WHEN itemid = 223835 THEN 
                CASE
                    WHEN valuenum > 0 AND valuenum <= 1 THEN valuenum * 100  -- Properly input data
                    WHEN valuenum > 1 AND valuenum < 21 THEN NULL  -- improperly input data - looks like O2 flow in liters
                    WHEN valuenum >= 21 AND valuenum <= 100 THEN valuenum  -- Within physiological range
                    ELSE NULL  -- Unphysiological values
                END
            WHEN itemid IN (3420, 3422) THEN valuenum  -- All these values are well formatted
            WHEN itemid = 190 AND valuenum > 0.20 AND valuenum < 1 THEN valuenum * 100  -- Well formatted but not in %
            ELSE valuenum
        END AS valuenum
    FROM
        chartevents
    WHERE
        itemid IN (211, 220045, 51, 442, 455, 6701, 220179, 220050, 225309, 8368, 8440, 8441, 8555, 220180, 
                    220051, 225310, 220052, 220181, 225312, 615, 618, 220210, 224690, 223761, 678, 223762, 
                    676, 646, 220277, 223835, 227444, 807, 811, 1529, 3745, 3744, 225664, 220621, 226537, 723, 
                    223900, 454, 223901, 184, 220739, 3420, 190, 223835, 3422, 52, 6702, 456)
)

SELECT
    icustay_id,
    charttime,
    itemid,
    label,
    valuenum
FROM
    labeled_data
WHERE
    label IS NOT NULL
    AND icustay_id IS NOT NULL
    AND valuenum IS NOT NULL
    AND charttime IS NOT NULL
ORDER BY
    icustay_id, charttime;
