import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#custom imports
from helper.utils import get_config, get_preproc_params

class Preprocessor:
    preproc_args = None
    config = None

    def __init__(self):
        self.tokenizer = get_tokenizer('basic_english')

    def start_preprocessing(self, data_iter):
        self.vocab = build_vocab_from_iterator(self.__yield_tokens(data_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.__build_text_pipeline()

    def collate_batch(self, batch):
        label_list, text_list, id_list, offsets = [], [], [], [0]

        for sample in batch:
            label_list.append(self.label_pipeline(sample["label"]))
            id_list.append(sample["id"])
            preprocessed_text = torch.tensor(self.text_pipeline(sample["text"]), dtype=torch.int64)
            text_list.append(preprocessed_text)
            offsets.append(preprocessed_text.size(0))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        id_list = torch.tensor(id_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list, text_list, id_list, offsets

    def get_vocab_size(self):
        return len(self.vocab)


    def __yield_tokens(self, data_iter):
        for text, _, _ in data_iter:
            yield self.tokenizer(text)

    def __build_text_pipeline(self):
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        self.label_pipeline = lambda x: int(x)



    


