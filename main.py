import model
from utils.data_utils import Season
from data.load_footystats import GenerateDataFrame
from data.load_csv import DataLoader

if __name__ == '__main__':
    season = Season.Past1
    df = GenerateDataFrame(season=season).load()
    # df = DataLoader('epl-training.csv', season).load()
    model.run(df)