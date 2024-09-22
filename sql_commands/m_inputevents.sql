DROP TABLE IF EXISTS m_inputevents;
CREATE TABLE m_inputevents AS

WITH input_cv AS (
    SELECT
        ie.icustay_id,
        ie.itemid,
        CASE 
            WHEN ie.itemid = 30047 THEN ie.rate / wt.weight
            WHEN ie.itemid = 30044 THEN ie.rate / wt.weight
            ELSE ie.rate
        END AS valuenum,
        ie.charttime,
        CASE
            WHEN ie.itemid IN (30120, 30047)        THEN 'rate_norepinephrine'
            WHEN ie.itemid IN (30119, 30309, 30044) THEN 'rate_epinephrine'
            WHEN ie.itemid IN (30043, 30307)        THEN 'rate_dopamine'
            WHEN ie.itemid IN (30042, 30306)        THEN 'rate_dobutamine'
            ELSE 'unknown'
        END AS label
    FROM inputevents_cv ie
    LEFT JOIN m_weight_first_day wt
        ON ie.icustay_id = wt.icustay_id
    WHERE ie.itemid IN (30120, 30119, 30309, 30043, 30307, 30042, 30306)
        AND ie.rate IS NOT NULL
),

input_mv AS (
    SELECT
        icustay_id,
        itemid,
        rate AS valuenum,
        starttime,
        endtime,
        CASE
            WHEN itemid IN (221906) THEN 'rate_norepinephrine'
            WHEN itemid IN (221289) THEN 'rate_epinephrine'
            WHEN itemid IN (221662) THEN 'rate_dopamine'
            WHEN itemid IN (221653) THEN 'rate_dobutamine'
            ELSE 'unknown'
        END AS label
    FROM inputevents_mv 
    WHERE itemid IN (221906, 221289, 221662, 221653)
    AND rate IS NOT NULL
),

time_series AS (
    SELECT
        icustay_id,
        itemid,
        label,
        valuenum,
        generate_series(starttime, endtime, interval '30 minutes') AS charttime
    FROM input_mv
)

-- Union der beiden Teilabfragen
SELECT
    i.icustay_id,
    i.itemid,
    i.label,
    i.valuenum,
    i.charttime
FROM input_cv i
INNER JOIN m_subjects ON i.icustay_id = m_subjects.icustay_id

UNION ALL

SELECT
    ts.icustay_id,
    ts.itemid,
    ts.label,
    ts.valuenum,
    ts.charttime
FROM time_series ts
INNER JOIN m_subjects ON ts.icustay_id = m_subjects.icustay_id
;