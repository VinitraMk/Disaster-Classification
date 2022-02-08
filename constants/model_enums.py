import enum

class Model(str, enum.Enum):
    ANN = 'ann'
    CNN = 'cnn'
    RNN = 'rnn'