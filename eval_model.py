import torch
import pickle

model_path = '.'
validation_file = '.'
features_file = '.'


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
        captions = []
        for _, x in image_source[start:end]:
            captions.extend(x['caption'])
        yield features, captions

def main():
    model = torch.load(model_path)

    with open(validation_file) as f:
        validation = pickle.load(f)
    features = scipy.io.loadmat(features_file)

    for x, y in make_mini_batch(features, validation):
        print x, y


if __name__ == '__main__':
    main()
