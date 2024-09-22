import pandas as pd
import os 
from .func_project_dir import *

def get_units():
    chartevents             = pd.read_csv(os.path.join(project_path(), "data/extracted_data/chartevents.csv"), index_col=0)
    labevents               = pd.read_csv(os.path.join(project_path(),"data/extracted_data/labevents.csv"), index_col=0)
    d_items                 = pd.read_csv(os.path.join(project_path(), "data/extracted_data/d_items.csv"), index_col=0)
    lab_measuring_units     = pd.read_csv(os.path.join(project_path(), "data/extracted_data/lab_measuring_units.csv"), index_col=0)

    map_units_lab = labevents.drop_duplicates(subset="ITEMID").merge(lab_measuring_units, on="ITEMID")
    map_units_lab = map_units_lab.sort_values(by="VALUEUOM")
    map_units_lab = map_units_lab.drop_duplicates(subset="LABEL", keep="first").reset_index()
    map_units_lab = map_units_lab[["LABEL", "VALUEUOM"]].rename(columns={"VALUEUOM":"UNIT"})

    map_units_chart = chartevents.drop_duplicates(subset="ITEMID").merge(d_items, on="ITEMID")
    map_units_chart = map_units_chart.sort_values(by="UNITNAME")
    map_units_chart = map_units_chart.drop_duplicates(subset="LABEL_x", keep="first").reset_index()
    map_units_chart = map_units_chart[["LABEL_x", "UNITNAME"]].rename(columns={"UNITNAME":"UNIT", "LABEL_x":"LABEL"})

    map_units       = pd.concat([map_units_chart, map_units_lab]).reset_index(drop=True)

    def clean_units(unit, label):
        if pd.isna(unit) or unit.lower().strip() == 'no unit':
            if 'GCS' in label:
                return 'Score'
            elif 'ptt' in label.lower() or 'pt' in label.lower():
                return 'sec'
            elif 'inr' in label.lower():
                return 'Ratio'
            elif 'temp' in label.lower():
                return '°Celsius'
            elif 'fio2' in label.lower():
                return '%'
            elif 'ph_bloodgas' in label.lower():
                return 'pH Units'
            elif 'glucose' in label.lower():
                return 'mg/dL'
            else:
                return 'Value'  # Allgemeine Standard-Einheit, falls keine andere passt
        
        unit = unit.lower().strip()
        
        # Korrekturen und Standardisierungen
        unit_mappings = {
            '%': '%',
            'bpm': 'bpm',
            'insp/min': 'breaths/min',
            'mmhg': 'mmHg',
            'iu/l': 'IU/L',
            'k/ul': 'K/uL',
            'mg/dl': 'mg/dL',
            'mm hg': 'mmHg',
            'seconds': 'sec',
            'units': 'pH Units',
            'mg/g': 'mg/g',
            'mmol/l': 'mmol/L',
            'ng/ml': 'ng/mL',
            'pg/ml': 'pg/mL',
            'sec': 'sec',
            '#/cu mm': '#/cu mm',
            'mm/hr': 'mm/hr',
            'mg/dl': 'mg/dL',
            '?c': '°C',
            'celsius': '°C',
            'mg/dl': 'mg/dL'
        }
        
        return unit_mappings.get(unit, unit)

    # Anwendung der Bereinigungsfunktion auf die UNIT-Spalte
    map_units['CLEANED_UNIT'] = map_units.apply(lambda row: clean_units(row['UNIT'], row['LABEL']), axis=1)
    map_units.drop(columns="UNIT", inplace=True)
    map_units = map_units.set_index("LABEL").to_dict()["CLEANED_UNIT"]
    return map_units