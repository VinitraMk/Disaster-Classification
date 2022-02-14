import torch.nn as nn
from torch.nn import Conv1d, ReLU, MaxPool1d, Sigmoid, EmbeddingBag, Linear, Dropout, Embedding
import math
import torch

from helper.utils import get_model_params


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
        self.dropout = Dropout(0.5)
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