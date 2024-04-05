import math
import pandas as pd
import numpy as np
from datetime import datetime

from data.utils import Season
from data.load_csv import DataLoader
from data.process import DataProcessor
from features.pi_rating import PiRatingsCalculator, RatingsManager
from features.individual_stats import IndividualTeamStats
from features.pairwise_stats import PairwiseTeamStats
from pipeline.X_table_constructor import XTableConstructor

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
    pi, pairwise_pi, weighted_pairwise_pi = RatingsManager(df).compute()

    print('Lenght of pi:', len(pi))
    print('Lenght of pairwise_pi:', len(pairwise_pi))
    print('Lenght of weighted_pairwise_pi:', len(weighted_pairwise_pi))
    # print(weighted_pairwise_pi.head())
    # sample_weighted_pi = weighted_pairwise_pi.loc[(weighted_pairwise_pi['HomeTeam'] == 'Man United') & (weighted_pairwise_pi['AwayTeam'] == 'Man City') | (weighted_pairwise_pi['HomeTeam'] == 'Man City') & (weighted_pairwise_pi['AwayTeam'] == 'Man United')].sort_index(ascending=False)
    # print(sample_weighted_pi)

def example_pairwise_stats(df: pd.DataFrame, unique_teams: list, individual_stats: pd.DataFrame):
    pairwise_stats = PairwiseTeamStats(df, unique_teams, individual_stats).compute()

    print(pairwise_stats.head())

    # print where HomeTeam = Southampton and AwayTeam = Brighton
    sample_pairwise_stats = pairwise_stats.loc[(pairwise_stats['HomeTeam'] == 'Southampton') & (pairwise_stats['AwayTeam'] == 'Brighton')]
    print(sample_pairwise_stats)

    # print the opposite
    sample_pairwise_stats = pairwise_stats.loc[(pairwise_stats['HomeTeam'] == 'Brighton') & (pairwise_stats['AwayTeam'] == 'Southampton')]
    print(sample_pairwise_stats)

def example_team_stats(df: pd.DataFrame, unique_teams: list):
    team_stats = IndividualTeamStats(df, unique_teams).compute()

    print(team_stats.head())
    return team_stats
    
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

    # individual_stats = example_team_stats(df, unique_teams)
    # example_pi(df)
    # example_pairwise_stats(df, unique_teams, individual_stats)
    pi_ratings, pi_pairwise, pi_weighted = RatingsManager(df).compute()
    individual_stats = IndividualTeamStats(df, unique_teams).compute()
    pairwise_stats = PairwiseTeamStats(df, unique_teams, individual_stats).compute()
    X_table = XTableConstructor(individual_stats, pairwise_stats, pi_ratings, pi_pairwise, pi_weighted).construct_table()
    print(X_table.shape)
    print(X_table.head())