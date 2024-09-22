import shutil
import warnings
warnings.simplefilter('ignore')

def consolidate_data():
    copy_list = [
                (r"./data/temp_data/df_input_time_imputed.csv", r"./data/target_data/df_input_time.csv"),
                (r"./data/temp_data/df_labevents_time_imputed.csv", r"./data/target_data/df_labevents_time.csv"),
                (r"./data/temp_data/df_vent_time_imputed.csv", r"./data/target_data/df_vent_time.csv"),
                (r"./data/temp_data/df_chartevents_time_imputed.csv", r"./data/target_data/df_chartevents_time.csv"),
                (r"./data/temp_data/df_output_time_imputed.csv", r"./data/target_data/df_output_time.csv"),
                (r"./data/temp_data/df_time_ranges_reduced.csv", r"./data/target_data/df_time_ranges.csv"),
                (r"./data/temp_data/df_subjects_reduced.csv", r"./data/target_data/df_subjects.csv"),
                (r"./data/temp_data/df_susp_reduced.csv", r"./data/target_data/df_susp.csv")
            ]

    for i in copy_list:
        shutil.copy(i[0], i[1])

def main():
    consolidate_data()
    
if __name__ == '__main__':
    main()