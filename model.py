import math
import pandas as pd
import numpy as np
from datetime import datetime

from data.utils import Season
from data.load_csv import DataLoader
from data.process import DataProcessor
from features.pi_rating import PiRatingsCalculator, PiRatingsManager
from features.individual_stats import IndividualTeamStats
from features.pairwise_stats import PairwiseTeamStats
from pipeline.X_table_constructor import XTrainConstructor, XTestConstructor
from pipeline.pre_processer import XTableEncoder, YSeriesEncoder, CrossChecker

import xgboost as xgb
from category_encoders import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import shap

def train_model(model, name, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series=None):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if name == 'CatBoost':
        y_pred = [int(x) for x in y_pred]

    # k-fold cross validation
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(model, X_train, y_train, cv=kfold)

    print(name + ':')
    print(y_pred)

    if y_test is None:
        return
    
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred, average='macro'))
    print('Precision:', precision_score(y_test, y_pred, average='macro'))
    print('Recall:', recall_score(y_test, y_pred, average='macro'))
    print(f'Cross Validation Accuracy: mean={round(results.mean(), 5)}, std={round(results.std(), 5)}')
    print()
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_train)
    # shap.initjs()
    # shap.summary_plot(shap_values, X_train, plot_type='bar')

    return y_pred

def read_test_csv():
    final_test_df = pd.read_csv('epl-test.csv')
    final_test_df = final_test_df.dropna(how='all')

    # Convert date to datetime
    for index, row in final_test_df.iterrows():
        date = row['Date'].split('-')
        if len(date[-1]) == 2:
            final_test_df.at[index, 'Date'] = f'{date[0]}/{date[1]}/20{date[-1]}'

    # Match team names to the ones used in the training set
    final_test_df = final_test_df.replace("Spurs", "Tottenham")
    final_test_df = final_test_df.replace("Nottingham Forest", "Nott'm Forest")
    final_test_df = final_test_df.replace("AFC Bournemouth", "Bournemouth")
    final_test_df = final_test_df.replace("Man Utd", "Man United")
    final_test_df = final_test_df.replace("Luton Town", "Luton")
    final_test_df = final_test_df.replace("Sheff Utd", "Sheffield United")

    final_test_df['Date'] = pd.to_datetime(final_test_df['Date'], format='mixed')
    final_test_df = final_test_df.sort_values(by=['Date'])
    return final_test_df

if __name__ == '__main__':
    dataset_path = 'epl-training.csv'
    season = Season.Past1

    df = DataLoader(dataset_path, season).load()
    data_processor = DataProcessor(df)
    unique_teams = data_processor.get_unique_teams()
    # df_train, df_test = data_processor.split_data(train_test_ratio=0.95)
    df_train, df_test = data_processor.split_data_last_n(n=10)
    # df_test = read_test_csv()
    print(len(df_train), len(df_test))

    X_train = XTrainConstructor(df_train, unique_teams, is_pairwise_stats=False, is_pi_ratings=True, is_pi_pairwise=False, is_pi_weighted=False).construct_table()
    y_train = df_train['FTR']
    X_train = XTableEncoder(X_train).run()
    y_train = YSeriesEncoder(y_train).run()

    X_test = XTestConstructor(df_test, df_train, unique_teams, is_pairwise_stats=False, is_pi_ratings=True, is_pi_pairwise=False, is_pi_weighted=False).construct_table()
    y_test = df_test['FTR']
    X_test = XTableEncoder(X_test).run()
    y_test = YSeriesEncoder(y_test).run()

    X_train, X_test = CrossChecker(X_train, X_test).run()

    # do cross validation
    # y_test = None
    log_reg = LogisticRegression(max_iter=1000)
    log_reg = train_model(log_reg, 'Logistic Regression', X_train, y_train, X_test, y_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    rf = train_model(rf, 'Random Forest', X_train, y_train, X_test, y_test)

    xgboost = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
    xgboost = train_model(xgboost, 'XGBoost', X_train, y_train, X_test, y_test)

    catboost = CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.1, loss_function='MultiClass')
    catboost.set_params(logging_level='Silent')
    catboost = train_model(catboost, 'CatBoost', X_train, y_train, X_test, y_test)