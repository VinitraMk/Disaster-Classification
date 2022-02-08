import torch
from torch.utils.data.dataset import Dataset
import os
import pandas as pd
from constants.model_enums import Model

from helper.utils import get_model_params

class DisasterDataset(Dataset):

    def __init__(self,input_path, is_train = True, transform = None):
        self.root_dir = os.environ["ROOT_DIR"]
        self.transform = transform
        self.data = pd.read_csv(input_path)
        if is_train:
            self.X = self.data.drop(columns=['target'])
            self.y = self.data['target']
        else:
            self.X = self.data
            self.y = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        model_args = get_model_params()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.y is not None:
            if model_args["model"] == Model.CNN or model_args["model"] == Model.RNN:
                label = self.y.iloc[idx]
            elif model_args["model"] == Model.ANN:
                label = [0.0000, 0.0000]
                if self.y.iloc[idx] == 1:
                    label[1] = 1.0000
                else:
                    label[0] = 1.0000
        else:
            if model_args["model"] == Model.CNN or model_args["model"] == Model.RNN:
                label = 0
            elif model_args == Model.ANN:
                label = [0.0000, 0.0000]
                

        #label = self.y.iloc[idx] if self.y is not None else -1
        text = self.X.iloc[idx]['text']
        id = self.X.iloc[idx]['id']
        sample = { "text": text, "label": label, "id": id }

        if self.transform:
            sample = self.transform(sample)
        return sample

        

    
