import sys
import importlib
sys.path.append('./py_scripts')

def run_modules():
    print("Running python modules...")
    module_names = [
        "00_set_project_path",
        "01_make_tables",
        "02_extract_data",
        "03_remove_outliers", 
        "04_data_preprocessing",
        "05_data_exclusion",
        "06_handle_missing_data",
        "07_sepsis_flagging",
        "08_stjohn_flagging",
        "09_consolidate_data",
        "10_ml_data_preparation",
        "11_create_features",
        "12_run_modelling"
    ]

    for i, module in enumerate(module_names):
        print(f"Running module {i+1}/{len(module_names)}: {module}")
        importlib.import_module(module).main()
        print(f"Module done: {module}")
        print("-"*100)

    print("Processing completed.")

def main():
    run_modules()


if __name__ == "__main__":
    main()



