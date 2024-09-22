-- source: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/concepts_postgres

-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
DROP TABLE IF EXISTS echo_data;
CREATE TABLE echo_data AS 
-- This code extracts structured data from echocardiographies
-- You can join it to the text notes using ROW_ID
-- Just note that ROW_ID will differ across versions of MIMIC-III.

SELECT ROW_ID,
       subject_id,
       hadm_id,
       chartdate,
       TO_TIMESTAMP(
           TO_CHAR(chartdate, 'YYYY-MM-DD') 
           || ' ' 
           || SUBSTRING(ne.text FROM 'Date/Time: .+? at ([0-9]+:[0-9]{2})') 
           || ':00',
           'YYYY-MM-DD HH24:MI:SS'
       ) AS charttime,
       SUBSTRING(ne.text FROM 'Indication: (.*?)\n') AS Indication,
       CAST(SUBSTRING(ne.text FROM 'Height: \(in\) ([0-9]+)') AS NUMERIC) AS Height,
       CAST(SUBSTRING(ne.text FROM 'Weight \(lb\): ([0-9]+)\n') AS NUMERIC) AS Weight,
       CAST(SUBSTRING(ne.text FROM 'BSA \(m2\): ([0-9]+) m2\n') AS NUMERIC) AS BSA,
       SUBSTRING(ne.text FROM 'BP \(mm Hg\): (.+)\n') AS BP,
       CAST(SUBSTRING(ne.text FROM 'BP \(mm Hg\): ([0-9]+)/[0-9]+?\n') AS NUMERIC) AS BPSys,
       CAST(SUBSTRING(ne.text FROM 'BP \(mm Hg\): [0-9]+/([0-9]+?)\n') AS NUMERIC) AS BPDias,
       CAST(SUBSTRING(ne.text FROM 'HR \(bpm\): ([0-9]+?)\n') AS NUMERIC) AS HR,
       SUBSTRING(ne.text FROM 'Status: (.*?)\n') AS Status,
       SUBSTRING(ne.text FROM 'Test: (.*?)\n') AS Test,
       SUBSTRING(ne.text FROM 'Doppler: (.*?)\n') AS Doppler,
       SUBSTRING(ne.text FROM 'Contrast: (.*?)\n') AS Contrast,
       SUBSTRING(ne.text FROM 'Technical Quality: (.*?)\n') AS TechnicalQuality
FROM noteevents ne
WHERE category = 'Echo';
