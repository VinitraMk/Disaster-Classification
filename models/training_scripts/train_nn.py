import sys
from pandas.core.indexing import is_label_like
import torch.nn as nn
from torch.nn import Conv1d, ReLU, MaxPool1d, Sigmoid, EmbeddingBag, Linear, Dropout, Embedding
import math
import torch
import torch
import os
import json

class CNN(nn.ModuleList):

    def __init__(self, vocab_size, embed_dim, num_classes, out_size, stride, text_max_len, final_output_size):
        super(CNN, self).__init__()
        self.embedding_size = embed_dim
        self.num_words = vocab_size
        self.seq_len = text_max_len
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        #self.kernel_4 = 5
        self.out_size = out_size 
        self.stride = stride

        #defining layers
        self.dropout = Dropout(0.25)
        self.embedding = Embedding(num_embeddings=self.num_words, embedding_dim=self.embedding_size, sparse = False, padding_idx=0)
        self.conv1 = Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.conv2 = Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv3 = Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
        #self.conv4 = Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)
        self.pool1 = MaxPool1d(self.kernel_1, self.stride)
        self.pool2 = MaxPool1d(self.kernel_2, self.stride)
        self.pool3 = MaxPool1d(self.kernel_3, self.stride)
        #self.pool4 = MaxPool1d(self.kernel_4, self.stride)
        self.relu = ReLU()
        self.fc = Linear(self.in_features_fc(), final_output_size)
        self.act = Sigmoid()

    def in_features_fc(self):
        '''Calculates the number of output features after Convolution + Max pooling
        Convolved_Features = ((embed_dim + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embed_dim + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        
        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''

        # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)
        
        # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)
        
        # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)
        
        # Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
        #out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        #out_conv_4 = math.floor(out_conv_4)
        #out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        #out_pool_4 = math.floor(out_pool_4)
      
        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3) * self.out_size

    def forward(self, text):
        embedded = self.embedding(text)
        #embedded = embedded.flatten()

        #convolution layer 1
        x1 = self.conv1(embedded)
        x1 = self.relu(x1)
        x1 = self.pool1(x1)

        #convolution layer 2
        x2 = self.conv2(embedded)
        x2 = self.relu(x2)
        x2 = self.pool2(x2)

        #convolution layer 1
        x3 = self.conv3(embedded)
        x3 = self.relu(x3)
        x3 = self.pool3(x3)

        #convolution layer 1
        #x4 = self.conv4(embedded)
        #x4 = self.relu(x4)
        #x4 = self.pool4(x4)

        output_union = torch.cat((x1, x2, x3), 2)
        output_union = output_union.reshape(output_union.size(0), -1)

        out = self.fc(output_union)
        out = self.dropout(out)
        return self.act(out).squeeze()

class ANN(nn.ModuleList):

    def __init__(self, vocab_size, embed_dim, num_classes):
        super(ANN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse = True)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.fc(embedded)
        return self.act(embedded)


class RNN(nn.Module):

    def __init__(self, lstm_size, embedding_dim, num_layers, num_words, dropout):
        super(RNN, self).__init__()
        self.lstm_size = lstm_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim = embedding_dim)
        self.lstm = nn.LSTM(input_size = self.lstm_size, hidden_size = self.lstm_size, num_layers = self.num_layers, dropout = dropout)
        self.fc = nn.Linear = nn.Linear(self.lstm_size, num_words)

    def forward(self, x, prev_state):
        embedded = self.embedding(x)
        output, state = self.lstm(embedded, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size), torch.zeros(self.num_layers, sequence_length, self.lstm_size))


class ExperimentTrain:

    def __init__(self, mounted_input_path, mounted_output_path, model_name, vocab_size, is_first_batch, is_last_batch, device, model_args):
        self.mounted_input_path = mounted_input_path
        self.mounted_output_path = mounted_output_path
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.is_first_batch = is_first_batch
        self.is_last_batch = is_last_batch
        self.device = device
        self.model_args = model_args
        self.model_details = dict()
        self.prev_state_h = None
        self.prev_state_c = None

    def __load_data_and_model(self):
        data_input_path = f"{self.mounted_input_path}/input"
        text_tensor_path = f"{data_input_path}/tensor_texts.pt"
        label_tensor_path = f"{data_input_path}/tensor_labels.pt"
        offset_tensor_path = f"{data_input_path}/tensor_offsets.pt"

        self.text_tensor = torch.load(text_tensor_path, map_location=self.device)
        self.label_tensor = torch.load(label_tensor_path, map_location=self.device)
        self.offset_tensor = torch.load(offset_tensor_path, map_location=self.device)
        self.text_tensor.to(self.device)
        self.label_tensor.to(self.device)
        self.offset_tensor.to(self.device)
        
        if self.is_first_batch:
            model_input_path = f"{self.mounted_input_path}/models/{self.model_args['model']}_model.tar"
        else:
            model_input_path = f"{self.mounted_output_path}/internal_output/{self.model_args['model']}_model.tar"
            model_details_path = f"{self.mounted_output_path}/internal_output/model_details.json"
            if os.path.exists(model_details_path):
                with open(model_details_path) as f:
                    self.model_details = json.load(f)

        self.model = CNN(self.vocab_size, self.model_args["embed_dim"], 2, model_args["out_size"], model_args["stride"], model_args["text_max_length"], model_args["final_output_size"])
        model_object = torch.load(model_input_path, map_location=self.device)
        self.model.load_state_dict(model_object["model_state"])
        self.model.to(self.device)
        if self.model_args["model"] == 'rnn':
            self.prev_state_h = model_object["rnn_prev_state_h"]
            self.prev_state_c = model_object["rnn_prev_state_c"]

        self.criterion = model_object["criterion"]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.model_args["lr"], momentum = self.model_args["momentum"])
        self.optimizer.load_state_dict(model_object["optimizer_state"])

    def __train_batch(self):
        self.model.train()
        self.optimizer.zero_grad()
        if self.model_args["model"] == 'ann':
            predicted_labels = self.model(self.text_tensor, self.offset_tensor)
        elif self.model_args["model"] == 'cnn':
            predicted_labels = self.model(self.text_tensor)
        elif self.model_args["model"] == 'rnn':
            predicted_labels, (state_h, state_c) = self.model(self.text_tensor, (self.prev_state_h, self.prev_state_c))
            self.prev_state_h, self.prev_state_c = state_h.detach(), state_c.detach()
        else:
            print('\nInvalid model name...')
            print('Exiting program...')
        
        self.label_tensor = self.label_tensor.type(torch.FloatTensor).to(self.device)
        loss = self.criterion(predicted_labels.squeeze(), self.label_tensor)
        print('actual output', predicted_labels[:5])
        print('expected outupt', self.label_tensor[:5])
        print('loss', loss)
        if self.is_first_batch:
            self.model_details["loss"] = loss.item()
        else:
            self.model_details["loss"] += loss.item()
        loss.backward()
        self.optimizer.step()
        model_object = {
            'model_state': self.model.state_dict(),
            'criterion': self.criterion,
            'optimizer_state': self.optimizer.state_dict()
        }

        if self.model_args["model"] == 'rnn':
            model_object["rnn_prev_state_h"], model_object["rnn_prev_state_c"] = self.prev_state_h, self.prev_state_c

        if not(self.is_last_batch):
            model_output_path = f"{self.mounted_output_path}/internal_output/{self.model_args['model']}_model.tar"
        else:
            model_output_path = f"{self.mounted_output_path}/models/{self.model_args['model']}_model.tar"

        if not(os.path.exists(f"{self.mounted_output_path}/internal_output")):
            os.mkdir(f"{self.mounted_output_path}/internal_output")

        if not(os.path.exists(f"{self.mounted_output_path}/models")):
            os.mkdir(f"{self.mounted_output_path}/models")
        print('Saving model...')
        torch.save(model_object, model_output_path)
        print('Saving model details...')
        with open(f"{self.mounted_output_path}/internal_output/model_details.json", "w+") as f:
            json.dump(self.model_details, f)

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
    model_args_string = sys.argv[7]

    model_args = json.loads(model_args_string)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        exit()
    exp = ExperimentTrain(mounted_input_path, mounted_output_path, model_name, vocab_size, is_first_batch, is_last_batch, device, model_args)
    exp.start_experiment()