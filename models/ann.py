import torch.nn as nn


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
