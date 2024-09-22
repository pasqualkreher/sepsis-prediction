-- Modified --> source: https://github.com/alistairewj/sepsis3-mimic 

DROP TABLE IF EXISTS m_outputevents; 
CREATE TABLE m_outputevents AS

WITH output_cte AS (
    SELECT
        oe.icustay_id,
        oe.itemid,
        oe.charttime,
        CASE 
            WHEN oe.value > 0 AND oe.value < 100000 THEN oe.value 
        END AS valuenum,
        'urine_output' AS label
    FROM outputevents oe
    WHERE oe.itemid IN (
        -- these are the most frequently occurring urine output observations in CareVue
        40055, -- "Urine Out Foley"
        43175, -- "Urine ."
        40069, -- "Urine Out Void"
        40094, -- "Urine Out Condom Cath"
        40715, -- "Urine Out Suprapubic"
        40473, -- "Urine Out IleoConduit"
        40085, -- "Urine Out Incontinent"
        40057, -- "Urine Out Rt Nephrostomy"
        40056, -- "Urine Out Lt Nephrostomy"
        40405, -- "Urine Out Other"
        40428, -- "Urine Out Straight Cath"
        40086, -- "Urine Out Incontinent"
        40096, -- "Urine Out Ureteral Stent #1"
        40651, -- "Urine Out Ureteral Stent #2"
        -- these are the most frequently occurring urine output observations in Metavision
        226559, -- "Foley"
        226560, -- "Void"
        227510, -- "TF Residual"
        226561, -- "Condom Cath"
        226584, -- "Ileoconduit"
        226563, -- "Suprapubic"
        226564, -- "R Nephrostomy"
        226565, -- "L Nephrostomy"
        226567, -- "Straight Cath"
        226557, -- "R Ureteral Stent"
        226558  -- "L Ureteral Stent"
    )
)
SELECT
    output_cte.icustay_id,
    output_cte.itemid,
    output_cte.charttime,
    output_cte.valuenum,
    output_cte.label
FROM output_cte
INNER JOIN m_subjects ON output_cte.icustay_id = m_subjects.icustay_id;

