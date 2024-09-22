from tqdm import tqdm
import sys
import os
base_path = os.getcwd() 
sys.path.append(base_path)  
from py_funcs.func_query_mimic import *

# Runs SQL commands from folder "sql_commands" to create tables
def make_tables():
    """
    Create tables in the PostgreSQL database
    """
    commands = [
            './sql_commands/m_subjects.sql',
            './sql_commands/m_echo_data.sql',
            './sql_commands/m_weight_first_day.sql',
            './sql_commands/m_inputevents.sql',
            './sql_commands/m_labevents.sql',
            './sql_commands/m_chartevents.sql',
            './sql_commands/m_outputevents.sql',
            './sql_commands/m_ventilation-durations.sql',
            './sql_commands/m_culture.sql',
            './sql_commands/m_antibiotics_pres.sql',
            './sql_commands/m_antibiotics_input.sql'
            ]
    
    conn = mimicdb().connect()
    cursor = conn.cursor()
    
    sections = len(commands)  # Total number of sections
    progress_bar = tqdm(total=sections, desc="Progress", position=0)

    for i, command in enumerate(commands):
        with open(fr"{command}", "r") as file:
            sql = file.read()
        try:
            cursor.execute(sql)
            progress_bar.update(1)
        except Exception as e:
            print(e)
            print(f"error in: {commands[i]}")
            break
    cursor.close()
    conn.commit()
    conn.close()

def main():
    make_tables()
    
if __name__ == '__main__':
    main()