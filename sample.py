import torch
#import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torch.autograd import Variable 
from torch import nn
from torchvision import transforms, models
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

class AlexNet2(nn.Module):

    def __init__(self, alexnet):
        super(AlexNet2, self).__init__()

        self.features = alexnet.features

        other_layer = list(alexnet.classifier.children())
        (self.d1, self.fc6, self.relu1,
         self.d2, self.fc7, self.relu2, self.fc8) = other_layer
        self.fc7_value = None
        self.fc6_value = None

    def forward(self, x):
        content = self.features(x)

        content = content.view(content.size(0), 256 * 6 * 6)
        contentn = self.d1(content)
        self.fc6_value = self.fc6(content)
        self.fc6_value = self.relu1(self.fc6_value)
        content = self.d2(self.fc6_value)
        self.fc7_value = self.fc7(content)
        self.fc7_value = self.relu2(self.fc7_value)
        content = self.fc8(self.fc7_value)
        return content


def main(args):
    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.Scale(args.crop_size),  
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    alexnet = models.alexnet(pretrained=True)
    alexnet2 = AlexNet2(alexnet)
    # Build Models
    encoder = EncoderCNN(4096, args.embed_size)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare Image       
    image = Image.open(args.image)
    image_tensor = Variable(transform(image).unsqueeze(0))
    
    # Set initial states
    state = (Variable(torch.zeros(args.num_layers, 1, args.hidden_size)),
             Variable(torch.zeros(args.num_layers, 1, args.hidden_size)))
    
    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        alexnet2.cuda()
        state = [s.cuda() for s in state]
        image_tensor = image_tensor.cuda()
    
    # Generate caption from image
    alexnet2(image_tensor)
    feature = encoder(alexnet2.fc7_value)
    sampled_ids = decoder.sample(feature, state)
    sampled_ids = sampled_ids.cpu().data.numpy()
    
    # Decode word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out image and generated caption.
    print (sentence)
#    plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='./models/encoder-5-3000.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-5-3000.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for center cropping images')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
