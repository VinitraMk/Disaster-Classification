import torch
from torch.utils.data.dataset import Dataset
import os
import pandas as pd

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.y.iloc[idx] if self.y is not None else -1
        text = self.X.iloc[idx]['text']
        id = self.X.iloc[idx]['id']
        sample = { "text": text, "label": label, "id": id }

        if self.transform:
            sample = self.transform(sample)
        return sample

        

    
