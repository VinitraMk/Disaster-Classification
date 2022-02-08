from helper.utils import get_config, get_preproc_params
from imblearn.over_sampling import SMOTE
import nlpaug
import nlpaug.augmenter.word as naw
import pandas as pd
import os

class Augmenter:
    preproc_args = None
    train_X = None
    y = None
    train_texts = None
    augmenter = None

    def __init__(self, train_X, y):
        self.preproc_args = get_preproc_params()
        self.train_X = train_X
        self.y = y

def generate_new_data(train_texts):
    config = get_config()
    aug = naw.SynonymAug()
    column_rename = lambda x: x + 20000
    data = train_texts.tolist()
    new_data = aug.augment(data)
    new_data_df = pd.DataFrame(new_data,columns=['text'])
    new_data_df.index.rename("id", inplace=True)
    new_data_df['new_index'] = new_data_df.index.map(column_rename)
    new_data_df.reset_index(drop=True, inplace=True)
    new_data_df.rename(columns={'new_index': 'id'}, inplace=True)
    new_data_df.to_csv(f'{config["input_path"]}\\new_data.csv', mode='w+', index=True)