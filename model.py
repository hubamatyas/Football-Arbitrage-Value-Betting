import math
import pandas as pd
import numpy as np
from datetime import datetime

from utils.data_utils import Season
from utils.model_utils import Feature
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

def get_feature_params():
    return {
        Feature.GOAL_STATS.value: False,
        Feature.SHOOTING_STATS.value: False,
        Feature.RESULT.value: False,
        Feature.HOME_AWAY_RESULTS.value: False,
        Feature.CONCEDED_STATS.value: False,
        Feature.LAST_N_MATCHES.value: False,
        Feature.WIN_STREAK.value: False,
        Feature.PAIRWISE_STATS.value: False,
        Feature.PI_RATINGS.value: True,
        Feature.PI_PAIRWISE.value: False,
        Feature.PI_WEIGHTED.value: False
    }

def train_model(model, name, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series=None):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

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

if __name__ == '__main__':
    dataset_path = 'epl-training.csv'
    season = Season.Past1

    df = DataLoader(dataset_path, season).load()
    data_processor = DataProcessor(df)
    unique_teams = data_processor.get_unique_teams()
    # df_train, df_test = data_processor.split_data(train_test_ratio=0.95)
    train, test = data_processor.split_data_last_n(n=10)
    feature_params = get_feature_params()

    X_train = XTrainConstructor(train.X, unique_teams, **feature_params).construct_table()
    X_train = XTableEncoder(X_train).run()
    y_train = YSeriesEncoder(train.y).run()

    X_test = XTestConstructor(test.X, train.X, unique_teams, **feature_params).construct_table()
    X_test = XTableEncoder(X_test).run()
    y_test = YSeriesEncoder(test.y).run()

    X_train, X_test = CrossChecker(X_train, X_test).run()

    # y_test = None
    lr = LogisticRegression(max_iter=1000)
    lr = train_model(lr, 'Logistic Regression', X_train, y_train, X_test, y_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    rf = train_model(rf, 'Random Forest', X_train, y_train, X_test, y_test)

    xgboost = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
    xgboost = train_model(xgboost, 'XGBoost', X_train, y_train, X_test, y_test)

    catboost = CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.1, loss_function='MultiClass')
    catboost.set_params(logging_level='Silent')
    catboost = train_model(catboost, 'CatBoost', X_train, y_train, X_test, y_test)