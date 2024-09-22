from tqdm import tqdm
import sys
import os
base_path = os.getcwd() 
sys.path.append(base_path)  
from py_funcs.func_query_mimic import *

# Script extract the created tables from the psql mimic-iii-database
# as .csv in to "./data/extracted_data"
def extract_data():
    chart = """
    SELECT v.*
    FROM  
    m_chartevents v 
    INNER JOIN m_subjects s ON v.icustay_id = s.icustay_id
    ;
    """
    lab = """
    SELECT l.*
    FROM  
    m_labevents l 
    INNER JOIN m_subjects s ON l.hadm_id = s.hadm_id
    ;
    """
    input = """
    SELECT i.*
    FROM  
    m_inputevents i
    INNER JOIN m_subjects s ON i.icustay_id = s.icustay_id
    ;
    """
    output = """
    SELECT o.*
    FROM  
    m_outputevents o 
    INNER JOIN m_subjects s ON o.icustay_id = s.icustay_id
    ;
    """
    vent = """
    SELECT v.*
    FROM  
    m_ventsettings v
    INNER JOIN m_subjects s ON v.icustay_id = s.icustay_id
    ;
    """
    culture = """
    SELECT m.*
    FROM  
    m_culture m
    INNER JOIN m_subjects s ON m.hadm_id = s.hadm_id
    ;
    """
    anti_pres = """
    SELECT a.*
    FROM  
    m_antibiotics_pres a
    INNER JOIN m_subjects s ON a.icustay_id = s.icustay_id
    ;
    """
    anti_input = """
    SELECT a.*
    FROM  
    m_antibiotics_input a
    INNER JOIN m_subjects s ON a.icustay_id = s.icustay_id
    ;
    """
    subjects = """
    SELECT s.*
    FROM  
    m_subjects s
    ;
    """
    d_items = """
    SELECT *
    FROM 
    d_items
    """

    lab_measuring_units = """
    SELECT DISTINCT 
    ITEMID, VALUEUOM 
    FROM labevents;
    """

    commands = {
                "chartevents":chart, 
                "labevents":lab, 
                "output":output, 
                "input":input,
                "vent":vent, 
                "subjects":subjects, 
                "culture":culture,
                "anti_pres":anti_pres,
                "anti_input":anti_input,
                "d_items":d_items,
                "lab_measuring_units":lab_measuring_units
            }
    
    sections = len(commands)  # Total number of sections
    progress_bar = tqdm(total=sections, desc="Progress", position=0)

    for command in commands:
        df = mimicdb().make_query(commands[command])
        df.to_csv(fr"./data/extracted_data/{command}.csv")
        progress_bar.update(1)

def main():
    extract_data()
    
if __name__ == '__main__':
    main()