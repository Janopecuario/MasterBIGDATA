## ----setup, include=FALSE------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(
	message = FALSE,
	error = TRUE,
	warning = FALSE,
	include = TRUE)
library(rmdformats)
library(prettydoc)
library(hrbrthemes)
library(tint)
library(tufte)


## import warnings
## import numpy as np
## import pandas as pd
## import matplotlib.pyplot as plt
## import seaborn as sns
## import sklearn
## from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
## from sklearn.model_selection import train_test_split, GridSearchCV
## from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
## import sys

## cars = pd.read_excel('datos_tarea25.xlsx')

## na_counts =cars.isnull().sum()
## print(na_counts[na_counts > 0])
## print(cars.duplicated().any())
## cars["Color"].value_counts()

## print(cars.head(4).to_string(index=False))
## cars.dtypes

## cars["Mileage"] = cars["Mileage"].str.replace(' km', ' ', regex=True).astype(int)
## ##Retiramos los caracteres y la convertimos en numérica
## cars["Levy"] = pd.to_numeric(cars["Levy"],errors ="coerce")
## #Con Coerce se asignan Na.
## cars["Engine volume"] = cars["Engine volume"].str.extract(r"([\d.]+)").astype(float) #ELiminación de la característica turbo y la pasamos a numérica
## cars["Cylinders"]= cars["Cylinders"].astype(object)
## #Conversión de Cylinders a categórica
## cars.dtypes

## pd.set_option("display.max_rows", None)
## pd.set_option("display.max_columns", None)
## pd.set_option("display.width", None)  # evita saltos de línea
## pd.set_option("display.max_colwidth", None)

## cars.describe().T

## cars.plot(
##     kind="box",
##     subplots=True,
##     layout=(4,3),
##     figsize=(4,3),
##     sharex=False,
##     sharey=False
## )
## plt.show()

## # warnings.filterwarnings("ignore")
## # for col in cars.select_dtypes(include=[np.number]).columns:
## #     Q1 = cars[col].quantile(0.25)
## #     Q3 = cars[col].quantile(0.75)
## #     IQR = Q3 - Q1
## #     lower = Q1 - 1.5 * IQR
## #     upper = Q3 + 1.5 * IQR
## #     # Reemplazar con  Nas
## #     cars.loc[(cars[col] < lower) | (cars[col] > upper), col] = np.nan
## # for col in cars.select_dtypes(include=[np.number]).columns:
## #     cars[col].fillna(cars[col].median(), inplace=True)
## # na_counts =cars.isnull().sum()
## # print(na_counts[na_counts > 0])

## cars["Cylinders"] = cars["Cylinders"].apply(lambda x: 4 if x == 4 else "Otro")
## categoricas = cars.select_dtypes(include=["object", "category"]).columns.tolist()

## categoricas = [x for x in categoricas if x not in ["Color"]]
## cars = pd.get_dummies(cars,columns=categoricas,drop_first=True)
## X=cars.drop(columns=["Color"])
## y= cars["Color"]
## X_train , X_test , y_train , y_test = train_test_split(X, y, test_size =0.2, random_state =1234)
## print(f'Frecuencia en train: \n{y_train.value_counts(normalize=True)}')
## print(f'\nFrecuencia en test: \n{y_test.value_counts(normalize=True)}')

## params0 = {
##     'max_depth': [3, 5, 10],
##     'min_samples_leaf': [15, 30,50],
##     'min_samples_split': [30,50, 75,100],
##     'criterion': ["gini", "entropy"]
## }
## scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
## grid_search0 = GridSearchCV(estimator=DecisionTreeClassifier(),
##                            param_grid=params0,
##                            cv=4, scoring = scoring_metrics, refit='accuracy')
## grid_search0.fit(X_train, y_train)

## results0 = pd.DataFrame(grid_search0.cv_results_).sort_values(by="mean_test_accuracy",ascending=False).reset_index(drop=True)
## params_df0 = pd.json_normalize(results0["params"])
## results0_expanded = pd.concat([params_df0, results0[['mean_test_accuracy', 'mean_test_precision_macro', 'mean_test_recall_macro', 'mean_test_f1_macro']]],axis=1)
## 
## results_abrev0 = results0_expanded.rename(columns={
##     "criterion": "criterion",
##     "max_depth": "depth",
##     "min_samples_leaf": "leaf",
##     "min_samples_split": "split",
##     "mean_test_accuracy": "accu",
##     "mean_test_precision_macro": "precision",
##     "mean_test_recall_macro": "recall",
##     "mean_test_f1_macro": "f1"
## })
## 
## results_abrev0 = results_abrev0.round(3)
## pd.set_option("display.max_rows", None)
## pd.set_option("display.max_columns", None)
## results_abrev0_5 = results_abrev0.iloc[::5].head(25)
## print(results_abrev0_5.head(7))
## print(results_abrev0.head(15))

## print(grid_search0.best_estimator_)

## params = {
##     'max_depth': list(range(6, 10)),
##     'min_samples_leaf': list(range(25, 40)),
##     'min_samples_split': list(range(25, 75)),
##     'criterion': ["gini", "entropy"]
## }
## scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
## grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),
##                            param_grid=params,
##                            cv=4, scoring = scoring_metrics, refit='accuracy')
## grid_search.fit(X_train, y_train)

## print(grid_search.best_estimator_)

## results = pd.DataFrame(grid_search.cv_results_).sort_values(by="mean_test_accuracy",ascending=False).reset_index(drop=True)
## params_df = pd.json_normalize(results["params"])
## results_expanded = pd.concat([params_df, results[['mean_test_accuracy', 'mean_test_precision_macro', 'mean_test_recall_macro', 'mean_test_f1_macro']]],axis=1)
## results_abrev = results_expanded.rename(columns={
##     "criterion": "criterion",
##     "max_depth": "depth",
##     "min_samples_split": "split",
##     "min_samples_leaf": "leaf",
##     "mean_test_accuracy": "accu",
##     "mean_test_precision_macro": "precision",
##     "mean_test_recall_macro": "recall",
##     "mean_test_f1_macro": "f1"
## })
## results_abrev = results_abrev.round(3)
## pd.set_option("display.max_rows", None)
## pd.set_option("display.max_columns", None)
## results_abrev_5 = results_abrev.iloc[::5].head(25)
## print(results_abrev_5.head(7))
## print(results_abrev.head(15))

## params = {
##     'max_depth': list(range(6, 10)),
##     #'min_samples_leaf': list(range(25, 40)),
##     'min_samples_split': list(range(25, 75)),
##     'criterion': ["gini", "entropy"]
## }
## scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
## grid_search2 = GridSearchCV(estimator=DecisionTreeClassifier(),
##                            param_grid=params,
##                            cv=4, scoring = scoring_metrics, refit='accuracy')
## grid_search2.fit(X_train, y_train)

## print(grid_search2.best_estimator_)

## results2 = pd.DataFrame(grid_search2.cv_results_).sort_values(by="mean_test_accuracy",ascending=False).reset_index(drop=True)
## params_df = pd.json_normalize(results2["params"])
## results_expanded2 = pd.concat([params_df, results2[['mean_test_accuracy', 'mean_test_precision_macro', 'mean_test_recall_macro', 'mean_test_f1_macro']]],axis=1)
## results_abrev2 = results_expanded2.rename(columns={
##     "criterion": "criterion",
##     "max_depth": "depth",
##     "min_samples_split": "split",
##     #"min_samples_leaf": "leaf",
##     "mean_test_accuracy": "accu",
##     "mean_test_precision_macro": "precision",
##     "mean_test_recall_macro": "recall",
##     "mean_test_f1_macro": "f1"
## })
## results_abrev2 = results_abrev2.round(3)
## pd.set_option("display.max_rows", None)
## pd.set_option("display.max_columns", None)
## results_abrev_52 = results_abrev2.iloc[::5].head(25)
## print(results_abrev_52.head(7))
## print(results_abrev2.head(15))

## cols = ['split0_test_accuracy','split1_test_accuracy','split2_test_accuracy','split3_test_accuracy']
## res_df = results2.loc[:15,cols]
## data = [row.values for _, row in res_df.iterrows()]
## labels = [f"res_{i}" for i in range(len(res_df))]
## plt.figure(figsize=(12,6))
## plt.boxplot(data, labels=labels)
## plt.title('Boxplots de Accuracy para los 4 Splits de 15 mejores modelos')
## plt.xlabel('Filas (res_i)')
## plt.ylabel('Accuracy')
## plt.xticks(rotation=45)
## plt.show()

## for i in range(15):  # de 0 a 14
##     params = results2['params'].iloc[i]
##     best_model = DecisionTreeClassifier(**params)
##     best_model.fit(X_train, y_train)
##     y_train_pred = best_model.predict(X_train)
##     y_test_pred = best_model.predict(X_test)
##     y_train_auc = pd.get_dummies(y_train, drop_first=True)
##     y_prob_train = best_model.predict_proba(X_train)[:, 1]
##     fpr, tpr, thresholds = roc_curve(y_train_auc, y_prob_train)
##     roc_auc_train = auc(fpr, tpr)
##     print(f"\nValor AUC {i} (train): {roc_auc_train:.2f}")
##     y_test_auc = pd.get_dummies(y_test, drop_first=True)
##     y_prob_test = best_model.predict_proba(X_test)[:, 1]
##     fpr, tpr, thresholds = roc_curve(y_test_auc, y_prob_test)
##     roc_auc_test = auc(fpr, tpr)
##     print(f"Valor AUC {i} (test): {roc_auc_test:.2f}")

## params = results2['params'].iloc[2]
## best_model = DecisionTreeClassifier(**params)
## tree_rules = export_text(best_model, feature_names=list(X_train.columns),show_weights=True)
## print(tree_rules)

## tree_rules = export_text(
##     best_model,
##     feature_names=list(X_train.columns),
##     show_weights=True,
##     max_depth=3)
## print(tree_rules)

## plt.figure(figsize=(20,10))
## plot_tree(best_model, feature_names=list(X_train.columns), class_names=best_model.classes_, filled=True)
## plt.show()

## plt.figure(figsize=(20,10))
## plot_tree(best_model, feature_names=list(X_train.columns), class_names=best_model.classes_, filled=True,max_depth=2)
## plt.show()

## df_importancia = pd.DataFrame({'Variable': best_model.feature_names_in_, 'Importancia': best_model.feature_importances_}).sort_values(by='Importancia', ascending=False)
## plt.bar(df_importancia['Variable'], df_importancia['Importancia'], color='skyblue')
## plt.xlabel('Variable')
## plt.ylabel('Importancia')
## plt.title('Importancia de las variables')
## plt.xticks(rotation=45, ha='right')
## plt.tight_layout()
## plt.show()

## plt.figure(figsize=(8, 6))
## plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
## plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
## plt.xlabel('1 - especificidad')
## plt.ylabel('Sensibilidad')
## plt.title('Curva ROC (Train)')
## plt.show()

## plt.figure(figsize=(8, 6))
## plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
## plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
## plt.xlabel("1 - especificidad")
## plt.ylabel('Sensibilidad')
## plt.title('Curva ROC con datos de test)')
## plt.show()

## from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
## RF_model = RandomForestClassifier(n_estimators = 60,bootstrap = True, max_depth = 8, min_samples_split=29, criterion='entropy',random_state=1234)
## RF_model.fit(X_train, y_train)
## y_pred_rf = RF_model.predict(X_test)
## accuracy_rf = accuracy_score(y_test, y_pred_rf)
## print(f'Precisión del modelo con Random Forest: {accuracy_rf}')

## params = {
##     'n_estimators' : [50,100,200,300,350],
##     'max_depth': list(range(6, 9)),,
##     'bootstrap': [True, False],
##     'min_samples_split': list(range(25, 75)),
##     'criterion': ["gini", "entropy"]
## }
## 
## scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
## 
## grid_search_RF = GridSearchCV(estimator=RF_model,
##                            param_grid=params,
##                            cv=4, scoring = scoring_metrics, refit='accuracy')
## grid_search_RF.fit(X_train, y_train)
## best_model_RF = grid_search_RF.best_estimator_
## print(grid_search_RF.best_estimator_)
## y_pred_rf = best_model_RF.predict(X_test)
## accuracy_rf_gs = accuracy_score(y_test, y_pred_rf)
## # Evaluar el rendimiento del modelo
## print(f'Precisión del árbol optimizado: {accuracy_rf_gs}')

## y_train_pred = best_model_RF.predict(X_train)
## y_test_pred = best_model_RF.predict(X_test)
## print(f'Se tiene un accuracy para train de: {accuracy_score(y_train,y_train_pred)}')
## print(f'Se tiene un accuracy para test de: {accuracy_score(y_test,y_test_pred)}')

## results_rf = pd.DataFrame(grid_search_RF.cv_results_).sort_values(by="mean_test_accuracy",ascending=False).reset_index(drop=True)
## params_df_rf = pd.json_normalize(results_rf["params"])
## results_expanded_rf = pd.concat([params_df_rf, results_rf[['mean_test_accuracy', 'mean_test_precision_macro', 'mean_test_recall_macro', 'mean_test_f1_macro']]],axis=1)
## results_abrev_rf = results_expanded_rf.rename(columns={
##     "criterion": "criterion",
##     "max_depth": "depth",
##     "min_samples_split": "split",
##     "mean_test_accuracy": "accu",
##     "mean_test_precision_macro": "precision",
##     "mean_test_recall_macro": "recall",
##     "mean_test_f1_macro": "f1"
## })
## results_abrev_rf = results_abrev_rf.round(3)
## pd.set_option("display.max_rows", None)
## pd.set_option("display.max_columns", None)
## results_abrev_5_rf = results_abrev_rf.iloc[::5].head(25)
## print(results_abrev_5_rf.head(7))
## print(results_abrev_rf.head(15))
## res_1 = results_rf[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[0]
## res_2 = results_rf[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[1]
## res_3 = results_rf[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[2]
## res_4 = results_rf[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[3]
## res_5 = results_rf[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[4]
## plt.boxplot([res_1.values,res_2.values,res_3.values,res_4.values,res_5.values], labels = ['res_1','res_2','res_3','res_4','res_5'])
## plt.title('Boxplots de Accuracy para los 4 Splits')
## plt.xlabel('Splits de Cross Validation')
## plt.ylabel('Accuracy')
## plt.show()

## import xgboost as xgb
## y_trainB = y_train.map({'White': 0, 'Black': 1})
## y_testB  = y_test.map({'White': 0, 'Black': 1})
## xgb_classifier = xgb.XGBClassifier(booster = 'gbtree', n_estimators = 400,
##                                eta = 0.1, gamma = 1, random_state=1234, max_depth = 9, tree_method = 'hist')
## xgb_classifier.fit(X_train, y_trainB)
## y_pred_baseB = xgb_classifier.predict(X_test)
## # Evaluar el rendimiento del modelo
## accuracy_a = accuracy_score(y_testB, y_pred_baseB)
## print(f'Precisión de gradient boosting: {accuracy_a}')

## params = {
##     'n_estimators': [100,200,300,400],
##     'eta' : [0.1,0.25,0.5,0.7],
##     'gamma' : [0.1,0.25,0.5,0.75,1],
##     'max_depth': [5,6,7,8,9]
## }
## scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
## grid_search_XGB = GridSearchCV(estimator=xgb_classifier,
##                            param_grid=params,
##                            cv=4, scoring = scoring_metrics, refit='accuracy')
## grid_search_XGB.fit(X_train, y_trainB)

## print(grid_search_XGB.best_estimator_.get_params)
## resultsXGB = pd.DataFrame(grid_search_XGB.cv_results_).sort_values(by="mean_test_accuracy",ascending=False).reset_index(drop=True)
## params_XGB = pd.json_normalize(resultsXGB["params"])
## resultsXGB_expanded = pd.concat([params_XGB, resultsXGB[['mean_test_accuracy', 'mean_test_precision_macro', 'mean_test_recall_macro', 'mean_test_f1_macro']]],axis=1)
## results_abrevXGB = resultsXGB_expanded.rename(columns={
##     "mean_test_precision_macro": "precision",
##     'mean_test_accuracy' : "acc",
##     "max_depth": "depth",
##     "mean_test_recall_macro": "recall",
##     "mean_test_f1_macro" : "f1"
##     })
## results_abrevXGB = results_abrevXGB.round(3)
## results_abrevXGB.head(15)
## sorted_results = resultsXGB.sort_values(by='mean_test_accuracy', ascending=True).head(5)
## res_1 = sorted_results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[0]
## res_2 = sorted_results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[1]
## res_3 = sorted_results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[2]
## res_4 = sorted_results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[3]
## res_5 = sorted_results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[4]
## plt.boxplot([res_1.values,res_2.values,res_3.values,res_4.values,res_5.values], labels = ['res_1','res_2','res_3','res_4','res_5'])
## plt.title('Boxplots de Accuracy para los 4 Splits')
## plt.xlabel('Splits de Cross Validation')
## plt.ylabel('Accuracy')
## plt.show()

## modelo_XGB = grid_search_XGB.best_estimator_
## y_train_pred_xgbB = modelo_XGB.predict(X_train)
## y_test_pred_xgbB = modelo_XGB.predict(X_test)
## print(f'Se tiene un accuracy para train de: {accuracy_score(y_trainB,y_train_pred_xgbB)}')
## print(f'Se tiene un accuracy para test de: {accuracy_score(y_testB,y_test_pred_xgbB)}')

## y_train_aucB = pd.get_dummies(y_trainB, drop_first=True)
## y_prob_trainB = modelo_XGB.predict_proba(X_train)[:, 1]
## fpr, tpr, thresholds = roc_curve(y_train_aucB, y_prob_trainB)
## roc_auc_train = auc(fpr, tpr)
## print(f"\nValor AUC (train): {roc_auc_train:.2f}")
## y_test_aucB = pd.get_dummies(y_testB, drop_first=True)
## y_prob_testB = modelo_XGB.predict_proba(X_test)[:, 1]
## fpr, tpr, thresholds = roc_curve(y_test_aucB, y_prob_testB)
## roc_auc_test = auc(fpr, tpr)
## print(f"Valor AUC (test Boosting): {roc_auc_test:.2f}")
