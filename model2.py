import model
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, vocab_size,
                 num_layer):
        super(Model, self).__init__()
        self.encoder = model.EncoderCNN(input_size, embedding_size)
        self.decoder = model.DecoderRNN(
            embedding_size, hidden_size, vocab_size,
            num_layer)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_feature, word_index, lengths):
        targets = pack_padded_sequence(word_index, lengths,
                                       batch_first=True)[0]
        features = self.encoder(image_feature)
        outputs = self.decoder(features, word_index, lengths)
        loss = self.criterion(outputs, targets)
        return loss






