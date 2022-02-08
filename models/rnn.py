from torch import lstm, nn
import torch

#custom imports


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

