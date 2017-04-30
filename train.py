import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

MODEL_PATH = 'modelpath'
vocab_path


def main(args):
    # Create model directory
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # Image preprocessing
#    transform = transforms.Compose([
#        transforms.RandomCrop(args.crop_size),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load vocabulary wrapper.
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
#    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
#                             transform, args.batch_size,
#                             shuffle=True, num_workers=args.num_workers)
#
    # Build the models
    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters())

    optimizer = torch.optim.Adam(params, lr=0.001)

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            # import pdb; pdb.set_trace()
            # Set mini-batch dataset
            images = Variable(images)
            captions = Variable(captions)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step,
                        loss.data[0], np.exp(loss.data[0])))

            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(),
                           os.path.join(args.model_path,
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(),
                           os.path.join(args.model_path,
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))
