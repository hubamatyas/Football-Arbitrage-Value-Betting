import math
import pandas as pd
import numpy as np
from datetime import datetime

from data.utils import Season
from data.load_csv import DataLoader
from data.process import DataProcessor
from features.pi_rating import PiRatingsCalculator, RatingsManager

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

if __name__ == '__main__':
    dataset_path = 'epl-training.csv'
    season = Season.Past5

    data_loader = DataLoader(dataset_path, season)
    data_loader.read_csv()
    data_loader.parse_df()
    data_loader.select_seasons(season)
    df = data_loader.get_df()
    print(df.head())
    data_processor = DataProcessor(df)
    print(len(data_processor.unique_teams))

    calculator = PiRatingsCalculator()
    manager = RatingsManager(df)
    manager.update_match_ratings(calculator)
    print("new")
    print(manager.pi_ratings.loc[(manager.pi_ratings['Team'] == 'Man United') | (manager.pi_ratings['Team'] == 'Man City')].sort_index(ascending=False))
    print(manager.pi_pairwise.loc[(manager.pi_pairwise['HomeTeam'] == 'Man United') & (manager.pi_pairwise['AwayTeam'] == 'Man City') | (manager.pi_pairwise['HomeTeam'] == 'Man City') & (manager.pi_pairwise['AwayTeam'] == 'Man United')].sort_index(ascending=False))
    
