import model
from utils.data_utils import Season

if __name__ == '__main__':
    dataset_path = 'epl-training.csv'
    season = Season.Past1
    model.run(dataset_path, season)