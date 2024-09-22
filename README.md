## Project Overview 
This project utilizes the MIMIC-III database to predict sepsis up to 12 hours before onset. Various machine learning models, such as Support Vector Machine, Logistic Regression, XGBoost, and Random Forest, are employed. Additionally, the rule-based St. John Sepsis Algorithm is implemented.The labels were created using sepsis labeling according to the Sepsis-3 criteria (SOFA Score).

The results of this study can be fully reproduced using the code provided in this GitHub repository. All steps, from data preprocessing and feature generation to model creation and evaluation, are documented and implemented in the corresponding scripts. The repository contains the complete code used to achieve the results presented in the study. Access to the MIMIC-III database requires appropriate permissions, which can be obtained through MIT. The data itself is not included in this repository; please refer to section 1.1 for instructions on how to access the MIMIC-III database.

## 1. Installation Instructions

### 1.1 Build MIMIC-III Database in PostgreSQL
- Follow these steps to set up the MIMIC-III database:
  - Get access to MIMIC-III: [MIMIC-III Access](https://physionet.org/content/mimiciii/1.4/)
  - Build the database: [MIMIC-III GitHub](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii)
  - For a helpful tutorial (especially for Windows): [YouTube Guide](https://www.youtube.com/watch?v=5rg1p7sg2Qo&t=315s&ab_channel=ObiedaAnanbeh)
  - **Important:** Update the `./files/mimic-iii-db.txt` file with the parameters set during your MIMIC-III setup.

### 1.2 Create a Virtual Environment
- Ensure you are using Python version `3.10.x`.
- Navigate to the project’s base folder, **mimic-iii-master**.
- Create and activate a virtual environment in this folder:
  ```bash
  python -m venv venv or python3 -m venv venv 
  On MAC/Linux: source ./venv/bin/activate  
  On Windows: .\venv\Scripts\activate

### 1.3 Run main.py
- This will:
  - Install dependencies from `requirements.txt` via pip.
  - Execute the `run_pyscripts.py` script.
  - **Note:** If issues arise, try installing the dependencies manually without specifying version numbers and/or downgrading or upgrading Python to version 3.10.11. You can also manually run `run_pyscripts.py`. If you encounter issues with the project path setup in `00_set_project_path.py`, set the path manually and comment out this script in `run_pyscripts.py`.

## 2. Overview of `run_pyscripts.py`
`run_pyscripts.py` coordinates a sequence of scripts, each playing a critical role in the MIMIC-III experiment setup.

### Scripts and Their Functions
- **00_set_project_path.py** Creates a new project_path.txt file with the current directory path written inside.
- **01_make_tables.py:** Initializes and creates tables in the database for data reduction.
- **02_extract_data.py:** Extracts data from the database as `.csv` for Python manipulation.
- **03_remove_outliers.py:** Removes outliers using the IQR method.
- **04_data_preprocessing.py:** Transforms data from long to wide format and calculates new dataframes.
- **05_data_exclusion.py:** Excludes ICU stays based on specific criteria.
- **06_handle_missing_data.py:** Handles missing data using interpolation, imputation, and forward/backward filling.
- **07_sepsis_flagging.py:** Flags sepsis cases using SOFA scores and calculates sepsis onset times.
- **08_stjohn_flagging.py:** Applies St. John’s rules to flag sepsis and severe sepsis cases.
- **09_consolidate_data.py:** Consolidates datasets from temporary directories to target directories.
- **10_ml_data_preparation.py:** Prepares balanced/unbalanced ML datasets by matching sepsis with non-sepsis cases.
- **11_create_features.py:** Generates ML features using lag statistics, clustering, and other calculations.
- **12_run_modelling.py:** Runs ML pipelines (e.g., Random Forest, XGBoost) to predict sepsis cases.

### Function Scripts
- **func_ml.py:** Holds functions for selecting data before Sepsis-Onset and provides train-test splits by ICUSTAY_ID.
- **func_query_mimic.py:** Provides an easy solution for querying the MIMIC database.
- **func_project_dir.py** Output the absolute project path.
- **func_units.py** Output a dict with the untis for Chartevents and Labevents.

## 3. Notebooks and Their Functions
- **feature_engineering.ipynb:** Visualizes the elbow method for selecting the number of clusters in admission diagnosis.
- **feature_exploration.ipynb:** Visualizes and analyzes key variables in the dataset to understand distributions and correlations.
- **feature_selection.ipynb:** Identifies the most relevant features, finds the optimal number of features, and selects the alpha for Lasso regression.
- **missing_data.ipynb:** Analyzes the imputation process.
- **ml_results.ipynb:** Evaluates and visualizes performance metrics (accuracy, precision, recall, etc.) of different machine learning models.
- **outliers.ipynb:** Analyzes outliers.
- **develope_modelling.ipynb:** Notebook from the develope process for `12_run_modelling.py`.
- **sample_size.ipynb.ipynb:** Information about size of samples used.

## 4. Note
The script `12_run_modelling.py` will need `lf_feature_selection.csv` and `rf_feature_selection.csv` from `./data/target_data/`. 
When you clone the repo you will get these by default. If you want to modify the selection consult the notebook `feature_selection.ipynb` in `./notebooks/`.