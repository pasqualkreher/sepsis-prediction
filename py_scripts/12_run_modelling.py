import time
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import warnings 
from scipy.stats import uniform, randint
import sys
import os
base_path = os.getcwd() 
sys.path.append(base_path)  
from py_funcs.func_ml import *
warnings.simplefilter('ignore')

def run_modeling():
    # Notice:
    # The comments are only written for the first Training Pipline, 
    # because all are very similar and vary not stronly

    ###########################################################################
    ### 1. Generate the train test ids, these are used in all Pipelines
    ###########################################################################
    ml_functions.make_train_test_ids(onset_hour=5, test_size=0.2) # you can vary with Spesis-Onset (onset_hour) and test size 
    df_train_ids    = pd.read_csv(r"./machine_learning/df_train_ids.csv", index_col=0)
    df_test_ids     = pd.read_csv(r"./machine_learning/df_test_ids.csv", index_col=0)


    # ###########################################################################
    # ### P1 First Pipeline with preselected Features
    # ###########################################################################
    # This pipline use the preselected features from the jupyter notebook
    # 'feature_selection.ipynb' in folder './notebooks' you can experiment
    # in the notebook, the inital one will be the one of this research 

    ###########################################################################
    ### P1.1 Loading the preselected features and the data
    ###########################################################################
    df_rf_feature_selection = pd.read_csv(r"./data/target_data/rf_feature_selection.csv", index_col=0)
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)
    df_ml = ml_functions.make_df_for_ml(df_ml, hours_before_sepsis_onset=12, hours_before_sepsis_cutoff=0)

    ###########################################################################
    ### P1.2 Make train and test dataframes
    ###########################################################################
    df_ml_train = df_ml.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner") 
    df_ml_test  = df_ml.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner")

    ###########################################################################
    ### P1.3 Make X-Train, -Test and y-Train, -Test 
    ###########################################################################
    X_train     = df_ml_train.iloc[:, 10:].loc[:, df_rf_feature_selection["Feature"]]
    y_train     = df_ml_train["SEPSIS_LABEL"]
    X_test      = df_ml_test.iloc[:, 10:].loc[:, df_rf_feature_selection["Feature"]]
    y_test      = df_ml_test["SEPSIS_LABEL"]

    ###########################################################################
    ### P1.4 Define Pipelines 
    ###########################################################################
    pipelines = {
        'RandomForest': Pipeline([
            ('model', RandomForestClassifier(random_state=42))
        ], verbose=True),

        'XGBClassifier': Pipeline([
            ('model', XGBClassifier(random_state=42))
        ], verbose=True),
    }

    ###########################################################################
    ### P1.5 Define Hyperparametergrid
    ###########################################################################
    param_grids = {
        'RandomForest': {
            'model__n_estimators': randint(10, 600),
            'model__max_depth': randint(1, 9),
            'model__min_samples_split': randint(2, 25),
            'model__min_samples_leaf': randint(1, 25),
            'model__max_features': ['sqrt', 'log2'],
        },
        'XGBClassifier': {
            'model__n_estimators': randint(10, 600),
            'model__max_depth': randint(1, 9),
            'model__learning_rate': uniform(0.01, 0.1),
            'model__subsample': uniform(0.5, 0.9),
        },
    }

    ###########################################################################
    ### P1.6 Initalize Gropkfold
    ###########################################################################
    gkf = GroupKFold(n_splits=20)
    groups = df_ml_train["ICUSTAY_ID"]

    ###########################################################################
    ### P1.7 Initalize Empty Result dict for saving scores, computetime and
    ###      parameters
    ###########################################################################
    results = {}

    ###########################################################################
    ### P1.8 Iterate over the pipelines and fit the models
    ###########################################################################
    # Perform RandomizedSearchCV for each pipeline
    for name, pipeline in pipelines.items():
        start_time = time.time() # Starttime used for claculate computation time
        # Use RandomSearchCV in with the groupkfold definied before
        search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], 
                                    n_iter=20, cv=gkf, random_state=42, n_jobs=-1, verbose=3)
        search.fit(X_train, y_train, groups=groups)
        end_time = time.time() # Endtime used for claculate computation time
        
        # Calculate socres accuracy, recall, precision
        best_model = search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        # Save the model as .joblib 
        model_filename = fr"./machine_learning/models/{name}_presel.joblib"
        joblib.dump(best_model, model_filename)
        
        # Store feature importances directly in the results dictionary
        feature_importances = None
        if hasattr(best_model.named_steps['model'], 'feature_importances_'):
            feature_importances = pd.Series(
                best_model.named_steps['model'].feature_importances_,
                index=X_train.columns
            ).to_dict() 

        # Store the results in the dictionary
        results[name] = {
            'model': name,
            'pipeline': 'preselected features',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'best_params': search.best_params_,
            'computation_time': end_time - start_time,
            'model_path': model_filename,
            'feature_importances': feature_importances,
            'selected_features': X_train.columns.tolist()
        }

        print(f"Best model for {name}: {search.best_params_}")
        print(f"Train Accuracy for {name}: {train_accuracy:.4f}")
        print(f"Test Accuracy for {name}: {test_accuracy:.4f}")
        print(f"Train Precision for {name}: {train_precision:.4f}")
        print(f"Test Precision for {name}: {test_precision:.4f}")
        print(f"Train Recall for {name}: {train_recall:.4f}")
        print(f"Test Recall for {name}: {test_recall:.4f}")
        print(f"Computation Time for {name}: {results[name]['computation_time']:.2f} seconds")

    # Save the results as a dataframe
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(r"./machine_learning/results/ml_results_presel.csv", index=False)


    ###########################################################################
    ### P2 Second Pipeline with all Features
    ###########################################################################
    # Random Forest and XGBoost - all features
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)
    df_ml = ml_functions.make_df_for_ml(df_ml, hours_before_sepsis_onset=12, hours_before_sepsis_cutoff=0)

    df_ml_train = df_ml.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner")
    df_ml_test = df_ml.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner")

    X_train = df_ml_train.iloc[:, 10:]
    y_train = df_ml_train["SEPSIS_LABEL"]
    X_test = df_ml_test.iloc[:, 10:]
    y_test = df_ml_test["SEPSIS_LABEL"]

    pipelines = {
        'RandomForest': Pipeline([
            ('model', RandomForestClassifier(random_state=42))
        ], verbose=True),

        'XGBClassifier': Pipeline([
            ('model', XGBClassifier(random_state=42))
        ], verbose=True),
    }

    param_grids = {
        'RandomForest': {
            'model__n_estimators': randint(10, 600),
            'model__max_depth': randint(1, 9),
            'model__min_samples_split': randint(2, 25),
            'model__min_samples_leaf': randint(1, 25),
            'model__max_features': ['sqrt', 'log2'],
        },
        'XGBClassifier': {
            'model__n_estimators': randint(10, 600),
            'model__max_depth': randint(1, 9),
            'model__learning_rate': uniform(0.01, 0.1),
            'model__subsample': uniform(0.5, 0.9),
        },
    }

    gkf = GroupKFold(n_splits=20)
    groups = df_ml_train["ICUSTAY_ID"]

    # Dictionary to store results
    results = {}

    # Perform RandomizedSearchCV for each pipeline
    for name, pipeline in pipelines.items():
        start_time = time.time()

        search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], 
                                    n_iter=20, cv=gkf, random_state=42, n_jobs=-1, verbose=3)
        search.fit(X_train, y_train, groups=groups)

        end_time = time.time()

        # Get the best model
        best_model = search.best_estimator_

        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        # Save the best model
        model_filename = fr"./machine_learning/models/{name}_all.joblib"
        joblib.dump(best_model, model_filename)

        # Extract feature importances if the model supports it
        feature_importances = None
        if hasattr(best_model.named_steps['model'], 'feature_importances_'):
            feature_importances = pd.Series(
                best_model.named_steps['model'].feature_importances_,
                index=X_train.columns
            ).to_dict()

        # Store the results in the dictionary
        results[name] = {
            'model': name,
            'pipeline': 'all features',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'best_params': search.best_params_,
            'computation_time': end_time - start_time,
            'model_path': model_filename,
            'feature_importances': feature_importances,
            'selected_features': X_train.columns.tolist()
        }

        print(f"Best model for {name}: {search.best_params_}")
        print(f"Train Accuracy for {name}: {train_accuracy:.4f}")
        print(f"Test Accuracy for {name}: {test_accuracy:.4f}")
        print(f"Train Precision for {name}: {train_precision:.4f}")
        print(f"Test Precision for {name}: {test_precision:.4f}")
        print(f"Train Recall for {name}: {train_recall:.4f}")
        print(f"Test Recall for {name}: {test_recall:.4f}")
        print(f"Computation Time for {name}: {results[name]['computation_time']:.2f} seconds")

    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(r"./machine_learning/results/ml_results_all.csv")


    ###########################################################################
    ### P3 Third pipeline with Random Forest, XGBoost, 
    ###    Logistic Regression and SVM - Standard Scaled + PCA
    ###########################################################################
    # Random Forest, XGBoost, Logistic Regression and SVM - Standard Scaled + PCA
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)
    df_ml = ml_functions.make_df_for_ml(df_ml, hours_before_sepsis_onset=12, hours_before_sepsis_cutoff=0)

    df_ml_train = df_ml.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"]) 
    df_ml_test  = df_ml.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"])

    X_train = df_ml_train.iloc[:, 10:]
    y_train = df_ml_train["SEPSIS_LABEL"]
    X_test  = df_ml_test.iloc[:, 10:]
    y_test  = df_ml_test["SEPSIS_LABEL"]


    pipelines = {
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('model', RandomForestClassifier(random_state=42))
        ]),

        'XGBClassifier': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('model', XGBClassifier(random_state=42))
        ]),

        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('model', LogisticRegression(random_state=42))
        ]),
        
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('model', SVC(probability=True, random_state=42, max_iter=1000))
        ])
    }

    # Define the parameter grids for each model
    param_grids = {
        'RandomForest': {
            'model__n_estimators': randint(10, 600),
            'model__max_depth': randint(1, 9),
            'model__min_samples_split': randint(2, 25),
            'model__min_samples_leaf': randint(2, 25),
            'model__max_features': ['sqrt', 'log2'],
        },
        'XGBClassifier': {
            'model__n_estimators': randint(10, 450),
            'model__max_depth': randint(1, 9),
            'model__learning_rate': uniform(0.01, 0.1),
            'model__subsample': uniform(0.5, 0.9),
        },
        'LogisticRegression': {
            'model__solver': ['liblinear', 'saga'],
            'model__penalty': ['l1', 'l2'],
            'model__C': uniform(0.01, 100),
        },
        'SVM': {
            'model__C': uniform(0.01, 10),
            'model__kernel': ['linear', 'rbf', 'poly'],
            'model__degree': randint(2, 5),
        }
    }


    gkf = GroupKFold(n_splits=20)
    groups = df_ml_train["ICUSTAY_ID"]

    # Dictionary to store results
    results = {}


    for name, pipeline in pipelines.items():
        start_time = time.time()
        
        search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], 
                                    n_iter=20, cv=gkf, random_state=42, n_jobs=-1, verbose=3)
        search.fit(X_train, y_train, groups=groups)
        
        end_time = time.time()
        
        # Get the best model
        best_model = search.best_estimator_
        
        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        # Save the best model
        model_filename = fr"./machine_learning/models/{name}_pca.joblib"
        joblib.dump(best_model, model_filename)
        
        # Get number of PCA components used
        if 'pca' in best_model.named_steps:
            n_components_used = best_model.named_steps['pca'].n_components_
        else:
            n_components_used = None
        
        # Store the results in the dictionary
        results[name] = {
            'model': name,
            'pipeline': 'pca, standard scaled',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'best_params': search.best_params_,
            'computation_time': end_time - start_time,
            'model_path': model_filename,
            'feature_importances': None,
            'selected_features': f"Number of Core Components: {n_components_used}"
        }
        
        print(f"Best model for {name}: {search.best_params_}")
        print(f"Train Accuracy for {name}: {train_accuracy:.4f}")
        print(f"Test Accuracy for {name}: {test_accuracy:.4f}")
        print(f"Train Precision for {name}: {train_precision:.4f}")
        print(f"Test Precision for {name}: {test_precision:.4f}")
        print(f"Train Recall for {name}: {train_recall:.4f}")
        print(f"Test Recall for {name}: {test_recall:.4f}")
        print(f"Computation Time for {name}: {results[name]['computation_time']:.2f} seconds")

    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(r"./machine_learning/results/ml_results_pca.csv")


    ###########################################################################
    ### P5 Fith Pipeline Logistic Regression and SVM - Standard Scaled + Lasso
    ###########################################################################
    # This pipline use the preselected alpha from the jupyter notebook
    # 'feature_selection.ipynb' in folder './notebooks' you can experiment
    # in the notebook, the inital one will be the one of this research 

    # Logistic Regression and SVM - Standard Scaled + Lasso
    # Lade die Daten
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)
    df_lr_feature_selection = pd.read_csv(r"./data/target_data/lf_feature_selection.csv", index_col=0)
    alpha = df_lr_feature_selection["Alpha"][0].item()
    df_ml = ml_functions.make_df_for_ml(df_ml, hours_before_sepsis_onset=12, hours_before_sepsis_cutoff=0)

    df_ml_train = df_ml.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"])
    df_ml_test = df_ml.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"])

    X_train = df_ml_train.iloc[:, 10:]
    y_train = df_ml_train["SEPSIS_LABEL"]
    X_test = df_ml_test.iloc[:, 10:]
    y_test = df_ml_test["SEPSIS_LABEL"]

    pipelines = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', SelectFromModel(Lasso(alpha=alpha, random_state=42))),
            ('model', LogisticRegression(random_state=42))
        ]),
        
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', SelectFromModel(Lasso(alpha=alpha, random_state=42))),
            ('model', SVC(probability=True, random_state=42, max_iter=1000))
        ])
    }

    # Definiere die Parametergrids für jedes Modell
    param_grids = {
        'LogisticRegression': {
            'model__solver': ['liblinear', 'saga'],
            'model__penalty': ['l1', 'l2'],
            'model__C': uniform(0.01, 100),
        },
        'SVM': {
            'model__C': uniform(0.1, 10),
            'model__kernel': ['linear', 'rbf', 'poly'],
            'model__degree': randint(2, 5),
        }
    }

    gkf = GroupKFold(n_splits=20)
    groups = df_ml_train["ICUSTAY_ID"]

    # Dictionary zum Speichern der Ergebnisse
    results = {}

    # Durchführen der RandomizedSearchCV für jede Pipeline
    for name, pipeline in pipelines.items():
        start_time = time.time()
        
        search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], 
                                    n_iter=20, cv=gkf, random_state=42, n_jobs=-1, verbose=3)
        search.fit(X_train, y_train, groups=groups)
        
        end_time = time.time()
        
        # Bestes Modell erhalten
        best_model = search.best_estimator_
        
        # Vorhersagen treffen
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Metriken berechnen
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        # Bestes Modell speichern
        model_filename = fr"./machine_learning/models/{name}_lasso_alpha_fix.joblib"
        joblib.dump(best_model, model_filename)
        
        # Feature-Selektionsschritt speichern
        selected_features = X_train.columns[best_model.named_steps['lasso'].get_support()]
        feature_importances = None
        if hasattr(best_model.named_steps['model'], 'coef_'):
            feature_importances = pd.Series(
                best_model.named_steps['model'].coef_[0],
                index=selected_features
            )
        
        # Ergebnisse im Dictionary speichern
        results[name] = {
            'model': name,
            'pipeline': 'standard scaled, lasso',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'best_params': search.best_params_,
            'computation_time': end_time - start_time,
            'model_path': model_filename,
            'feature_importances': feature_importances,
            'selected_features': X_train.columns.tolist()

        }
        
        print(f"Best model for {name}: {search.best_params_}")
        print(f"Train Accuracy for {name}: {train_accuracy:.4f}")
        print(f"Test Accuracy for {name}: {test_accuracy:.4f}")
        print(f"Train Precision for {name}: {train_precision:.4f}")
        print(f"Test Precision for {name}: {test_precision:.4f}")
        print(f"Train Recall for {name}: {train_recall:.4f}")
        print(f"Test Recall for {name}: {test_recall:.4f}")
        print(f"Computation Time for {name}: {results[name]['computation_time']:.2f} seconds")

    # Ergebnisse als CSV speichern
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(r"./machine_learning/results/ml_results_lasso_alpha_fix.csv")


    ###########################################################################
    ### P6 Sixth Pipeline Logistic Regression - 
    ###    Standard Scaled + Lasso searching Alpha 
    ###########################################################################
    # Logistic Regression - Standard Scaled + Lasso searching Alpha 
    # Lade die Daten
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)
    df_ml = ml_functions.make_df_for_ml(df_ml, hours_before_sepsis_onset=12, hours_before_sepsis_cutoff=0) 

    df_ml_train = df_ml.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"])
    df_ml_test = df_ml.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"])

    X_train = df_ml_train.iloc[:, 10:]
    y_train = df_ml_train["SEPSIS_LABEL"]
    X_test = df_ml_test.iloc[:, 10:]
    y_test = df_ml_test["SEPSIS_LABEL"]

    # Definiere die Pipeline für die Logistic Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', SelectFromModel(Lasso(random_state=42))),  
        ('model', LogisticRegression(random_state=42))
    ])

    # Parametergrid definieren
    param_grid = {
        'lasso__estimator__alpha': uniform(0.0001, 0.05),
        'model__solver': ['liblinear', 'saga'],
        'model__penalty': ['l1', 'l2'],
        'model__C': uniform(0.01, 100),
    }
    gkf = GroupKFold(n_splits=20)
    groups = df_ml_train["ICUSTAY_ID"]

    # Dictionary zum Speichern der Ergebnisse
    results = {}

    # RandomizedSearchCV für die Pipeline durchführen
    start_time = time.time()

    search = RandomizedSearchCV(pipeline, param_distributions=param_grid, 
                                n_iter=20, cv=gkf, random_state=42, n_jobs=-1, verbose=3)
    search.fit(X_train, y_train, groups=groups)

    end_time = time.time()

    # Bestes Modell erhalten
    best_model = search.best_estimator_

    # Vorhersagen treffen
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Metriken berechnen
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)

    # Bestes Modell speichern
    model_filename = r"./machine_learning/models/LogisticRegression_lasso_alpha_search.joblib"
    joblib.dump(best_model, model_filename)

    # Feature-Selektionsschritt speichern
    selected_features = X_train.columns[best_model.named_steps['lasso'].get_support()]
    feature_importances = None
    if hasattr(best_model.named_steps['model'], 'coef_'):
        # Koeffizienten als Feature-Importanzen verwenden
        feature_importances = pd.Series(
            best_model.named_steps['model'].coef_[0],
            index=selected_features
        )

    # Ergebnisse im Dictionary speichern
    results['LogisticRegression'] = {
        'model': 'LogisticRegression',
        'pipeline': 'standard scaled, lasso alpha_search',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'best_params': search.best_params_,
        'computation_time': end_time - start_time,
        'model_path': model_filename,
        'feature_importances': feature_importances,
        'selected_features': selected_features.tolist(),
    }

    print(f"Best model: {search.best_params_}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Train Precision: {train_precision:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Train Recall: {train_recall:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Computation Time: {results['LogisticRegression']['computation_time']:.2f} seconds")

    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(r"./machine_learning/results/ml_results_lasso_alpha_search.csv")


    ##########################################################################
    ## P7 Seventh Pipeline Stacked clf
    ##    Random Forest and XGBoost - Fitted model for each hour from 1 to 12
    ##########################################################################
    # This models are used to be stacked later on and are just trained for one
    # singel hour before Sepsis-Onset

    # Stacked clf
    # Random Forest and XGBoost - Fitted model for each hour from 1 to 12
    df_rf_feature_selection = pd.read_csv(r"./data/target_data/rf_feature_selection.csv", index_col=0)
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)

    hours = [i for i in range(1, 13)]
    for hour in hours:
        df_ml_hour = ml_functions.make_df_for_ml(df_ml, hours_before_sepsis_onset=hour, hours_before_sepsis_cutoff=hour-1)

        df_ml_train = df_ml_hour.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner") 
        df_ml_test = df_ml_hour.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner")

        X_train = df_ml_train.iloc[:, 10:].loc[:, df_rf_feature_selection["Feature"]]
        y_train = df_ml_train["SEPSIS_LABEL"]
        X_test = df_ml_test.iloc[:, 10:].loc[:, df_rf_feature_selection["Feature"]]
        y_test = df_ml_test["SEPSIS_LABEL"]

        pipelines = {
            'RandomForest': Pipeline([
                ('model', RandomForestClassifier(random_state=42))
            ], verbose=True),

            'XGBClassifier': Pipeline([
                ('model', XGBClassifier(random_state=42))
            ], verbose=True),
        }

        param_grids = {
            'RandomForest': {
                'model__n_estimators': randint(10, 600),
                'model__max_depth': randint(1, 9),
                'model__min_samples_split': randint(2, 25),
                'model__min_samples_leaf': randint(2, 25),
                'model__max_features': ['sqrt', 'log2'],
            },
            'XGBClassifier': {
                'model__n_estimators': randint(10, 450),
                'model__max_depth': randint(1, 9),
                'model__learning_rate': uniform(0.01, 0.1),
                'model__subsample': uniform(0.5, 0.9),
            },
        }

        gkf = GroupKFold(n_splits=25)
        groups = df_ml_train["ICUSTAY_ID"]

        # Perform RandomizedSearchCV for each pipeline
        for name, pipeline in pipelines.items():
            start_time = time.time()
            
            search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], 
                                        n_iter=20, cv=gkf, random_state=42, n_jobs=-1, verbose=3)
            search.fit(X_train, y_train, groups=groups)
            
            end_time = time.time()
            
            # Get the best model
            best_model = search.best_estimator_
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_precision = precision_score(y_train, y_train_pred)
            test_precision = precision_score(y_test, y_test_pred)
            train_recall = recall_score(y_train, y_train_pred)
            test_recall = recall_score(y_test, y_test_pred)
            
            # Save the best model
            model_filename = fr"./machine_learning/models/{name}_presel_hour_{hour}.joblib"
            joblib.dump(best_model, model_filename)

    ###########################################################################
    ### P8 Eight Pipeline XGBoost - Hours stacked clf
    ###########################################################################
    # Here will be the models of singel hours are stracked
    ## XGBoost - Hours stacked clf
    df_rf_feature_selection = pd.read_csv(r"./data/target_data/rf_feature_selection.csv", index_col=0)
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)
    df_ml = ml_functions.make_df_for_ml(df_ml, hours_before_sepsis_onset=12, hours_before_sepsis_cutoff=0)

    df_ml_train = df_ml.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner") 
    df_ml_test = df_ml.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner")

    X_train = df_ml_train.iloc[:, 10:].loc[:, df_rf_feature_selection["Feature"]]
    y_train = df_ml_train["SEPSIS_LABEL"]
    X_test = df_ml_test.iloc[:, 10:].loc[:, df_rf_feature_selection["Feature"]]
    y_test = df_ml_test["SEPSIS_LABEL"]

    xgb_hour_1 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_1.joblib")
    xgb_hour_2 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_2.joblib")
    xgb_hour_3 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_3.joblib")
    xgb_hour_4 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_4.joblib")
    xgb_hour_5 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_5.joblib")
    xgb_hour_6 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_6.joblib")
    xgb_hour_7 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_7.joblib")
    xgb_hour_8 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_8.joblib")
    xgb_hour_9 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_9.joblib")
    xgb_hour_10 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_10.joblib")
    xgb_hour_11 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_11.joblib")
    xgb_hour_12 = joblib.load(r"./machine_learning/models/XGBClassifier_presel_hour_12.joblib")

    base_clf = [
        ("xgb_hour_1", xgb_hour_1),
        ("xgb_hour_2", xgb_hour_2),
        ("xgb_hour_3", xgb_hour_3),
        ("xgb_hour_4", xgb_hour_4),
        ("xgb_hour_5", xgb_hour_5),
        ("xgb_hour_6", xgb_hour_6),
        ("xgb_hour_7", xgb_hour_7),
        ("xgb_hour_8", xgb_hour_8),
        ("xgb_hour_9", xgb_hour_9),
        ("xgb_hour_10", xgb_hour_10),
        ("xgb_hour_11", xgb_hour_11),
        ("xgb_hour_12", xgb_hour_12)
    ]

    meta_model  = LogisticRegression(random_state=42)
    stack       = StackingClassifier(estimators=base_clf, final_estimator=meta_model, cv=20, n_jobs=-1, verbose=3)
    start_time  = time.time()
    stack.fit(X_train, y_train)
    end_time = time.time()

    model_filename = r"./machine_learning/models/StackedXGB.joblib"
    joblib.dump(stack, model_filename)

    # Make predictions with the stack
    y_train_pred = stack.predict(X_train)
    y_test_pred = stack.predict(X_test)

    # Calculate metrics
    train_accuracy  = accuracy_score(y_train, y_train_pred)
    test_accuracy   = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision  = precision_score(y_test, y_test_pred)
    train_recall    = recall_score(y_train, y_train_pred)
    test_recall     = recall_score(y_test, y_test_pred)

    results = {}

    results['StackedXGB'] = {
        'model': 'StackedXGB',
        'pipeline': 'StackedXGB',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'best_params': stack.get_params(),
        'computation_time': end_time - start_time,
        'model_path': model_filename,
        'feature_importances': None,
        'selected_features': stack.feature_names_in_.tolist(),
    }

    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(r"./machine_learning/results/ml_results_StackedXGB.csv", index=False)

    ###########################################################################
    ### P9 Ninth Pipeline Random Forest - Hours stacked clf
    ###########################################################################
    # Here will be the models of singel hours are stracked
    ## Random Forest - Hours stacked clf
    df_rf_feature_selection = pd.read_csv(r"./data/target_data/rf_feature_selection.csv", index_col=0)
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)
    df_ml = ml_functions.make_df_for_ml(df_ml, hours_before_sepsis_onset=12, hours_before_sepsis_cutoff=0)

    df_ml_train = df_ml.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner") 
    df_ml_test = df_ml.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner")

    X_train = df_ml_train.iloc[:, 10:].loc[:, df_rf_feature_selection["Feature"]]
    y_train = df_ml_train["SEPSIS_LABEL"]
    X_test = df_ml_test.iloc[:, 10:].loc[:, df_rf_feature_selection["Feature"]]
    y_test = df_ml_test["SEPSIS_LABEL"]

    RF_hour_1 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_1.joblib")
    RF_hour_2 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_2.joblib")
    RF_hour_3 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_3.joblib")
    RF_hour_4 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_4.joblib")
    RF_hour_5 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_5.joblib")
    RF_hour_6 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_6.joblib")
    RF_hour_7 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_7.joblib")
    RF_hour_8 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_8.joblib")
    RF_hour_9 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_9.joblib")
    RF_hour_10 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_10.joblib")
    RF_hour_11 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_11.joblib")
    RF_hour_12 = joblib.load(r"./machine_learning/models/RandomForest_presel_hour_12.joblib")

    base_clf = [
        ("RF_hour_1", RF_hour_1),
        ("RF_hour_2", RF_hour_2),
        ("RF_hour_3", RF_hour_3),
        ("RF_hour_4", RF_hour_4),
        ("RF_hour_5", RF_hour_5),
        ("RF_hour_6", RF_hour_6),
        ("RF_hour_7", RF_hour_7),
        ("RF_hour_8", RF_hour_8),
        ("RF_hour_9", RF_hour_9),
        ("RF_hour_10", RF_hour_10),
        ("RF_hour_11", RF_hour_11),
        ("RF_hour_12", RF_hour_12)
    ]

    meta_model  = LogisticRegression(random_state=42)
    stack       = StackingClassifier(estimators=base_clf, final_estimator=meta_model, cv=20, n_jobs=-1, verbose=3)
    start_time  = time.time()
    stack.fit(X_train, y_train)
    end_time = time.time()

    model_filename = r"./machine_learning/models/StackedRF.joblib"
    joblib.dump(stack, model_filename)

    # Make predictions with the stack
    y_train_pred = stack.predict(X_train)
    y_test_pred = stack.predict(X_test)

    # Calculate metrics
    train_accuracy  = accuracy_score(y_train, y_train_pred)
    test_accuracy   = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision  = precision_score(y_test, y_test_pred)
    train_recall    = recall_score(y_train, y_train_pred)
    test_recall     = recall_score(y_test, y_test_pred)

    results = {}

    results['StackedRF'] = {
        'model': 'StackedRF',
        'pipeline': 'StackedRF',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'best_params': stack.get_params(),
        'computation_time': end_time - start_time,
        'model_path': model_filename,
        'feature_importances': None,
        'selected_features': stack.feature_names_in_.tolist(),
    }

    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(r"./machine_learning/results/ml_results_StackedRF.csv", index=False)


    ###########################################################################
    ### P10 Tenth Pipeline XGBoost and Logistic Regreesion - Stacked
    ###########################################################################
    ## XGBoost and Logistic Regreesion - Stacked
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)
    df_ml = ml_functions.make_df_for_ml(df_ml, hours_before_sepsis_onset=12, hours_before_sepsis_cutoff=0)

    df_ml_train = df_ml.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner") 
    df_ml_test = df_ml.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner")

    X_train = df_ml_train.iloc[:, 10:]
    y_train = df_ml_train["SEPSIS_LABEL"]
    X_test = df_ml_test.iloc[:, 10:]
    y_test = df_ml_test["SEPSIS_LABEL"]

    xgb = joblib.load(r"./machine_learning/models/XGBClassifier_all.joblib")
    lr = joblib.load(r"./machine_learning/models/LogisticRegression_lasso_alpha_search.joblib")

    base_clf = [
        ("xgb", xgb),
        ("lr", lr),
    ]

    meta_model  = LogisticRegression(random_state=42)
    stack       = StackingClassifier(estimators=base_clf, final_estimator=meta_model, cv=20, n_jobs=-1, verbose=3)
    start_time  = time.time()
    stack.fit(X_train, y_train)
    end_time = time.time()

    model_filename = r"./machine_learning/models/StackedXGB_LR.joblib"
    joblib.dump(stack, model_filename)

    # Make predictions with the stack
    y_train_pred = stack.predict(X_train)
    y_test_pred = stack.predict(X_test)

    # Calculate metrics
    train_accuracy  = accuracy_score(y_train, y_train_pred)
    test_accuracy   = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision  = precision_score(y_test, y_test_pred)
    train_recall    = recall_score(y_train, y_train_pred)
    test_recall     = recall_score(y_test, y_test_pred)

    results = {}

    results['StackedXGB_LR'] = {
        'model': 'StackedXGB_LR',
        'pipeline': 'StackedXGB_LR',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'best_params': stack.get_params(),
        'computation_time': end_time - start_time,
        'model_path': model_filename,
        'feature_importances': None,
        'selected_features': stack.feature_names_in_.tolist(),
    }

    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(r"./machine_learning/results/ml_results_StackedXGB_LR.csv", index=False)


    ###########################################################################
    ### P11 Eleventh Pipeline Stack stacked hours 
    ###     Random Forest and stacked hours XGBoost
    ###########################################################################
    ## Stack stacked hours Random Forest and stacked hours XGBoost
    df_rf_feature_selection = pd.read_csv(r"./data/target_data/rf_feature_selection.csv", index_col=0)
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)
    df_ml = ml_functions.make_df_for_ml(df_ml, hours_before_sepsis_onset=12, hours_before_sepsis_cutoff=0)

    df_ml_train = df_ml.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner") 
    df_ml_test = df_ml.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner")

    X_train = df_ml_train.iloc[:, 10:].loc[:, df_rf_feature_selection["Feature"]]
    y_train = df_ml_train["SEPSIS_LABEL"]
    X_test = df_ml_test.iloc[:, 10:].loc[:, df_rf_feature_selection["Feature"]]
    y_test = df_ml_test["SEPSIS_LABEL"]

    rf = joblib.load(r"./machine_learning/models/StackedRF.joblib")
    xgb = joblib.load(r"./machine_learning/models/StackedXGB.joblib")

    base_clf = [
        ("rf", rf),
        ("xgb", xgb),
    ]

    meta_model  = LogisticRegression(random_state=42)
    stack       = StackingClassifier(estimators=base_clf, final_estimator=meta_model, cv=20, n_jobs=-1, verbose=3)
    start_time  = time.time()
    stack.fit(X_train, y_train)
    end_time = time.time()

    model_filename = r"./machine_learning/models/StackedRF_XGB.joblib"
    joblib.dump(stack, model_filename)

    # Make predictions with the stack
    y_train_pred = stack.predict(X_train)
    y_test_pred = stack.predict(X_test)

    # Calculate metrics
    train_accuracy  = accuracy_score(y_train, y_train_pred)
    test_accuracy   = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision  = precision_score(y_test, y_test_pred)
    train_recall    = recall_score(y_train, y_train_pred)
    test_recall     = recall_score(y_test, y_test_pred)

    results = {}

    results['StackedRF_XGB'] = {
        'model': 'StackedRF_XGB',
        'pipeline': 'StackedRF_XGB',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'best_params': stack.get_params(),
        'computation_time': end_time - start_time,
        'model_path': model_filename,
        'feature_importances': None,
        'selected_features': stack.feature_names_in_.tolist(),
    }

    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(r"./machine_learning/results/ml_results_StackedRF_XGB.csv", index=False)

    ###########################################################################
    ### P12 Twelveth Pipeline Random Forest and XGBoost - 
    ###     Reference first hour no engineered features 
    ###########################################################################
    # Random Forest and XGBoost - Reference first hour no engineered features 
    df_ml = pd.read_csv(r"./data/target_data/df_ml.csv", index_col=0)
    df_ml = df_ml.groupby("ICUSTAY_ID").first().reset_index().iloc[:,:47]

    df_ml_train = df_ml.merge(df_train_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner") 
    df_ml_test = df_ml.merge(df_test_ids, on=["SEPSIS_ID", "RANDOM_NO_SEPSIS_ID"], how="inner")

    X_train = df_ml_train.iloc[:, 10:]
    y_train = df_ml_train["SEPSIS_LABEL"]
    X_test = df_ml_test.iloc[:, 10:]
    y_test = df_ml_test["SEPSIS_LABEL"]

    pipelines = {
        'RandomForest': Pipeline([
            ('model', RandomForestClassifier(random_state=42))
        ], verbose=True),

        'XGBClassifier': Pipeline([
            ('model', XGBClassifier(random_state=42))
        ], verbose=True),
    }

    param_grids = {
        'RandomForest': {
            'model__n_estimators': randint(10, 600),
            'model__max_depth': randint(1, 9),
            'model__min_samples_split': randint(2, 25),
            'model__min_samples_leaf': randint(1, 25),
            'model__max_features': ['sqrt', 'log2'],
        },
        'XGBClassifier': {
            'model__n_estimators': randint(10, 600),
            'model__max_depth': randint(1, 9),
            'model__learning_rate': uniform(0.01, 0.1),
            'model__subsample': uniform(0.5, 0.9),
        },
    }

    # Dictionary to store results
    results = {}

    # Perform RandomizedSearchCV for each pipeline
    for name, pipeline in pipelines.items():
        start_time = time.time()
        
        search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], 
                                    n_iter=20, cv=20, random_state=42, n_jobs=-1, verbose=3)
        search.fit(X_train, y_train)
        
        end_time = time.time()
        
        best_model = search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        model_filename = fr"./machine_learning/models/{name}_first_hour.joblib"
        joblib.dump(best_model, model_filename)
        
        # Store feature importances directly in the results dictionary
        feature_importances = None
        if hasattr(best_model.named_steps['model'], 'feature_importances_'):
            feature_importances = pd.Series(
                best_model.named_steps['model'].feature_importances_,
                index=X_train.columns
            ).to_dict() 

        # Store the results in the dictionary
        results[name] = {
            'model': name,
            'pipeline': 'first_hour',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'best_params': search.best_params_,
            'computation_time': end_time - start_time,
            'model_path': model_filename,
            'feature_importances': feature_importances,
            'selected_features': X_train.columns.tolist()
        }

        print(f"Best model for {name}: {search.best_params_}")
        print(f"Train Accuracy for {name}: {train_accuracy:.4f}")
        print(f"Test Accuracy for {name}: {test_accuracy:.4f}")
        print(f"Train Precision for {name}: {train_precision:.4f}")
        print(f"Test Precision for {name}: {test_precision:.4f}")
        print(f"Train Recall for {name}: {train_recall:.4f}")
        print(f"Test Recall for {name}: {test_recall:.4f}")
        print(f"Computation Time for {name}: {results[name]['computation_time']:.2f} seconds")

    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(r"./machine_learning/results/ml_results_first_hour.csv", index=False)

def main():
    run_modeling()

if __name__ == '__main__':
    main()