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

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series=None):
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('XGBoost:')

    if y_test is None:
        y_test = pd.DataFrame()
        y_test['FTR'] = y_pred
        y_test['FTR'] = y_test['FTR'].map({0: 'H', 1: 'A', 2: 'D'})
        print(y_test)
        return model
    
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred, average='macro'))
    print('Precision:', precision_score(y_test, y_pred, average='macro'))
    print('Recall:', recall_score(y_test, y_pred, average='macro'))
    print(y_pred)
    print()

    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series=None):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Random Forest:')

    if y_test is None:
        y_test = pd.DataFrame()
        y_test['FTR'] = y_pred
        y_test['FTR'] = y_test['FTR'].map({0: 'H', 1: 'A', 2: 'D'})
        print(y_test)
        return model
    
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred, average='macro'))
    print('Precision:', precision_score(y_test, y_pred, average='macro'))
    print('Recall:', recall_score(y_test, y_pred, average='macro'))
    print(y_pred)
    print()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.initjs()
    shap.summary_plot(shap_values, X_train, plot_type='bar')

    return model

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series=None):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Logistic Regression:')

    if y_test is None:
        y_test = pd.DataFrame()
        y_test['FTR'] = y_pred
        y_test['FTR'] = y_test['FTR'].map({0: 'H', 1: 'A', 2: 'D'})
        print(y_test)
        return model
    
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred, average='macro'))
    print('Precision:', precision_score(y_test, y_pred, average='macro'))
    print('Recall:', recall_score(y_test, y_pred, average='macro'))
    print(y_pred)
    print()

    return model

def train_catboost(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series=None):
    model = CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.1, loss_function='MultiClass')
    model.set_params(logging_level='Silent')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('CatBoost:')

    if y_test is None:
        y_pred = [int(x) for x in y_pred]
        y_test = pd.DataFrame()
        y_test['FTR'] = y_pred
        y_test['FTR'] = y_test['FTR'].map({0: 'H', 1: 'A', 2: 'D'})
        print(y_test)
        return model
    
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred, average='macro'))
    print('Precision:', precision_score(y_test, y_pred, average='macro'))
    print('Recall:', recall_score(y_test, y_pred, average='macro'))
    print(y_pred)
    print()

    return model

def read_test_csv():
    final_test_df = pd.read_csv('epl-test.csv')
    final_test_df = final_test_df.dropna(how='all')

    # Convert date to datetime
    for index, row in final_test_df.iterrows():
        date = row['Date'].split('-')
        # print(date[-1])
        if len(date[-1]) == 2:
            final_test_df.at[index, 'Date'] = f'{date[0]}/{date[1]}/20{date[-1]}'
            # print(final_test_df.at[index, 'Date'])

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
    model = train_random_forest(X_train, y_train, X_test, y_test)
    model = train_logistic_regression(X_train, y_train, X_test, y_test)
    model = train_xgboost(X_train, y_train, X_test, y_test)
    model = train_catboost(X_train, y_train, X_test, y_test)