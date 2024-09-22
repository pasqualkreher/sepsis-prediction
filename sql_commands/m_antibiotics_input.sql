DROP TABLE IF EXISTS m_antibiotics_input; 
CREATE TABLE m_antibiotics_input AS 

WITH inp_ct AS 
(   
    SELECT
        icustay_id,
        itemid,
        MIN(charttime) AS antibiotic_starttime,
        MAX(charttime) AS antibiotic_endtime
    FROM
        chartevents AS c
    WHERE
        itemid IN (
            303, 3369, 3445, 3554, 3679, 4215, 4220, 4252, 4256, 4364, 4415, 4426,
            4511, 4538, 4557, 4587, 4590, 4715, 4716, 4732, 4762, 4905, 4972, 5060,
            5062, 5063, 5157, 5204, 6201, 6258, 6381, 6536, 6610, 6975, 7362, 7612
        )
        AND icustay_id IS NOT NULL
    GROUP BY
        icustay_id,
        itemid
),

inp_mv AS
(
    SELECT
        icustay_id,
        imv.itemid,
        starttime AS antibiotic_starttime,
        endtime AS antibiotic_endtime
    FROM
        inputevents_mv AS imv
    INNER JOIN
        d_items AS di
        ON imv.itemid = di.itemid
    WHERE
        imv.itemid IN (
            225798, 225837, 225838, 225840, 225842, 225843, 225844, 225845,
            225847, 225848, 225850, 225851, 225853, 225855, 225857, 225859,
            225860, 225862, 225863, 225865, 225866, 225868, 225869, 225871,
            225873, 225875, 225876, 225877, 225879, 225881, 225882, 225883,
            225884, 225885, 225886, 225888, 225889, 225890, 225892, 225893,
            225895, 225896, 225897, 225898, 225899, 225900, 225902, 225903,
            225905, 227691, 228003
        )
),

inp AS
(
    SELECT * FROM inp_ct
    UNION ALL
    SELECT * FROM inp_mv
)

SELECT inp.icustay_id, inp.itemid, inp.antibiotic_starttime, inp.antibiotic_endtime
FROM inp
INNER JOIN m_subjects ON inp.icustay_id = m_subjects.icustay_id;
