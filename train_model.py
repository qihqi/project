import random
import json
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import scipy.io as si

from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

embed_dim = 256
hidden_size = 512
num_layers_rnn = 1
image_data_file = './image_data.pickle'
image_feature_file = './img_features.mat'
vocab_path = './data/vocab2.pkl'

def make_mini_batch(image_features, image_data, batch_size=32, use_caption=None):
    image_source = image_data.items()
    random.shuffle(image_source)
    if use_caption is None:
        use_caption = random.randint(0, 4)

    for i in range(0, len(image_source), batch_size):
        start = i
        end = i + batch_size
        if end >= len(image_source):
            end = len(image_source)
        feature_idx =[[x[1]['fid'] for x in
                     image_source[start:end]]]
        features = image_features[feature_idx]
        captions = [x['caption'][use_caption] for _, x in
                    image_source[start:end]]
        yield features, captions


def make_word_padding(sentences, vocab):
    '''takes a sentence and produce an np.array
       representing the indexes of the words in
       the vocab
       '''
    word_index = list(
        map(lambda x:
              [vocab('<start>')] +
              [vocab(i) for i in x.split()] +
              [vocab('<end>')]
            , sentences))
    length = len(word_index)

    tagged = sorted([(i, len(i)) for i in word_index],
                    key=lambda x: -x[1])
    max_length = tagged[0][1]

    words = []
    length = []
    for x, l in tagged:
        for _ in range(max_length - l):
            x.append(vocab('<pad>'))
        words.append(x)
        length.append(l)

    word_index = np.array(words)
    word_index = word_index.astype(np.int64)
    return word_index, length



def main():
    # Load vocabulary wrapper.
    with open(vocab_path) as f:
        vocab = pickle.load(f)

    encoder = EncoderCNN(4096, embed_dim)
    decoder = DecoderRNN(embed_dim, hidden_size,
                         len(vocab), num_layers_rnn)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    #load data
    with open(image_data_file) as f:
        image_data = pickle.load(f)
    image_features = si.loadmat(image_feature_file)

    img_features = image_features['fc7'][0]
    img_features = np.concatenate(img_features)

    print 'here'
    iteration = 0

    for i in range(10): # epoch
        use_caption = i % 5
        print 'Epoch', i
        for x, y in make_mini_batch(img_features, image_data,
                                    use_caption=use_caption):
            word_padding, lengths = make_word_padding(y, vocab)

            x = Variable(torch.from_numpy(x).cuda())
            word_index = Variable(torch.from_numpy(word_padding).cuda())

            encoder.zero_grad()
            decoder.zero_grad()

            features = encoder(x)
            targets = pack_padded_sequence(word_index, lengths,
                                           batch_first=True)[0]
            outputs = decoder(features, word_index, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print 'loss', loss.data[0]

            iteration += 1

        torch.save(decoder.state_dict(), 'decoder.pkl' )
        torch.save(encoder.state_dict(), 'encoder.pkl' )




if __name__ == '__main__':
    main()
