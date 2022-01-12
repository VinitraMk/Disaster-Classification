import sys
from pandas.core.indexing import is_label_like
import torch.nn as nn
import torch
import os

class ANN(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes):
        super(ANN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse = True)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.act = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.fc(embedded)
        return self.act(embedded)


class ExperimentTrain:

    def __init__(self, mounted_input_path, mounted_output_path, model_name, vocab_size, is_first_batch, is_last_batch):
        self.mounted_input_path = mounted_input_path
        self.mounted_output_path = mounted_output_path
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.is_first_batch = is_first_batch
        self.is_last_batch = is_last_batch

    def __load_data_and_model(self):
        data_input_path = f"{self.mounted_input_path}/input"
        text_tensor_path = f"{data_input_path}/tensor_texts.pt"
        label_tensor_path = f"{data_input_path}/tensor_labels.pt"
        offset_tensor_path = f"{data_input_path}/tensor_offsets.pt"

        self.text_tensor = torch.load(text_tensor_path)
        self.label_tensor = torch.load(label_tensor_path)
        self.offset_tensor = torch.load(offset_tensor_path)

        if self.is_first_batch:
            model_input_path = f"{self.mounted_input_path}/models/ann_model.pt"
        else:
            model_input_path = f"{self.mounted_output_path}/internal_models/ann_model.pt"

        self.model = ANN(self.vocab_size, 64, 2)
        model_object = torch.load(model_input_path)
        self.model.load_state_dict(model_object["model_state"])
        self.criterion = model_object["criterion"]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 1.0)
        self.optimizer.load_state_dict(model_object["optimizer_state"])

    def __train_batch(self):
        self.model.train()
        self.optimizer.zero_grad()
        predicted_labels = self.model(self.text_tensor, self.offset_tensor)
        print(predicted_labels[:5])
        print(self.label_tensor[:5])
        loss = self.criterion(predicted_labels, self.label_tensor)
        loss.backward()
        self.optimizer.step()
        model_object = {
            'model_state': self.model.state_dict(),
            'criterion': self.criterion,
            'optimizer_state': self.optimizer.state_dict()
        }

        if not(self.is_last_batch):
            model_output_path = f"{self.mounted_output_path}/internal_models/ann_model.pt"
        else:
            model_output_path = f"{self.mounted_output_path}/models/ann_model.pt"

        if not(os.path.exists(f"{self.mounted_output_path}/internal_models")):
            os.mkdir(f"{self.mounted_output_path}/internal_models")

        if not(os.path.exists(f"{self.mounted_output_path}/models")):
            os.mkdir(f"{self.mounted_output_path}/models")

        with open(model_output_path, "wb") as f:
            torch.save(model_object, f)



    def start_experiment(self):
        self.__load_data_and_model()
        self.__train_batch()





if __name__ == "__main__":

    mounted_input_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    model_name = sys.argv[3]
    vocab_size = int(sys.argv[4])
    is_first_batch = True if sys.argv[5] == "True" else False
    is_last_batch = True if sys.argv[6] == "True" else False
    print(sys.argv[5], type(sys.argv[5]), is_first_batch)

    exp = ExperimentTrain(mounted_input_path, mounted_output_path, model_name, vocab_size, is_first_batch, is_last_batch)
    exp.start_experiment()