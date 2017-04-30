

encoder_path = './encoder.pkl'
decoder_path = './decoder.pkl'

def get_model(encoder_path, decoder_path):
    encoder = EncoderCNN(4096, embed_dim)
    decoder = DecoderRNN(embed_dim, hidden_size,
                         len(vocab), num_layers_rnn)
def main():


