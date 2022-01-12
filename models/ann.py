import torch.nn as nn


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