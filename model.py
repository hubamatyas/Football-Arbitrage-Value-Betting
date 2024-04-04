import math
import pandas as pd
import numpy as np
from datetime import datetime

from data.utils import Season
from data.load_csv import DataLoader
from data.process import DataProcessor
from features.pi_rating import PiRatingsCalculator, RatingsManager
from features.descriptive_stats import DescriptiveStats, IndividualTeamStats

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

def example_pi(df: pd.DataFrame):
    calculator = PiRatingsCalculator()
    manager = RatingsManager(df)
    manager.update_match_ratings(calculator)

    weighted_pairwise_pi = manager.get_weighted_ratings()
    print(weighted_pairwise_pi.head())
    sample_weighted_pi = weighted_pairwise_pi.loc[(weighted_pairwise_pi['HomeTeam'] == 'Man United') & (weighted_pairwise_pi['AwayTeam'] == 'Man City') | (weighted_pairwise_pi['HomeTeam'] == 'Man City') & (weighted_pairwise_pi['AwayTeam'] == 'Man United')].sort_index(ascending=False)
    print(sample_weighted_pi)

def example_stats(df: pd.DataFrame):
    processor = DescriptiveStats(df, 6)
    team_scores = processor.compute_team_scores()
    stats = processor.compute_stats()
    features_df = processor.generate_features_dataframe(stats)
    print(features_df.head())

def example_team_stats(df: pd.DataFrame, unique_teams: list):
    team_stats = IndividualTeamStats(df, unique_teams)
    team_stats.compute_team_stats()
    team_stats_df = team_stats.generate_features_dataframe()
    print(team_stats_df.head())
    
if __name__ == '__main__':
    dataset_path = 'epl-training.csv'
    season = Season.Past5

    data_loader = DataLoader(dataset_path, season)
    data_loader.read_csv()
    data_loader.parse_df()
    data_loader.select_seasons(season)
    df = data_loader.get_df()
    data_processor = DataProcessor(df)
    unique_teams = data_processor.unique_teams

    example_team_stats(df, unique_teams)