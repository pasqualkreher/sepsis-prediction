DROP TABLE IF EXISTS m_culture; 
CREATE TABLE m_culture AS

SELECT hadm_id,
  CASE WHEN mic.charttime IS NOT NULL THEN mic.charttime ELSE mic.chartdate END AS culture_time,
  spec_itemid,
  max(case when org_name is not null and org_name != '' then 1 else 0 end) as positive_culture
FROM microbiologyevents as mic
GROUP BY hadm_id, culture_time, spec_itemid

--WHERE 
  --LOWER(mic.spec_type_desc) LIKE '%blood%'
--AND LOWER(mic.spec_type_desc) LIKE '%culture%';
