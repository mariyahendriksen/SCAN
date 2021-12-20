from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import logging
import argparse

from layers_resnest import Layers_resnest, transform_mlf


device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def save_list_to_txt(elements, txt_file):
    with open(txt_file, 'w') as f:
        for el in elements:
            f.write(f"{el}\n")
    print('Saved list to ', txt_file)

def load_list_from_txt(txt_file):
    with open(txt_file, 'r') as f:
        data = f.readlines()
        data = [line.strip() for line in data]
    return data

def get_image_features(model, root, file_paths):
    features = []
    for file_path in tqdm(file_paths):
        path = os.path.join(root, file_path)
        img = Image.open(path)
        img_transformed = transform_mlf(img).unsqueeze(0).to(device)
        feature = model.forward1(img_transformed).to('cpu').squeeze().detach().numpy()
        features.append(feature)

    features = np.stack(features, axis=0)
    return features


def main(args):
    root_raw = args.root_raw
    df_file = args.df_file
    split = args.split
    root_precomp = args.root_precomp
    logging.info(args)

    # 1. loads:
    # layers model
    net = Layers_resnest(img_dim=2048, trained_dresses=False, checkpoint_path=None)
    net.eval()
    logging.info('Loaded layers resnest')
    # df
    df = pd.read_csv(
        df_file,
        dtype={
            'image_name': str,
            'eval_status': str,
            'product_type': str
        },
        index_col=0
    )
    logging.info('Loaded df')
    df_subset = df[df['eval_status'] == split]
    file_paths = df_subset['image_name'].tolist()
    logging.info(f'Got image_names for {split}')

    # 2. embed images
    features = get_image_features(net, root_raw, file_paths)
    logging.info('Completed image transformations, final images shape: ', features.shape)

    # 3. save image representations
    target_file = os.path.join(root_precomp, f'mlf_{split}_ims.npy')
    with open(target_file, 'wb') as f:
        np.save(f, features)
    logging.info(f"Saved image file to {target_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_raw',
                        default='/Users/mhendriksen/Desktop/repositories/SCAN/data/deep_fashion/category-and-attribute-prediction',
                        help='path to image folder')
    parser.add_argument('--df_file',
                        default='/Users/mhendriksen/Desktop/repositories/SCAN/data/deep_fashion/category-and-attribute-prediction/eval_partition.csv',
                        help='df file path')
    parser.add_argument('--split',
                        default='train',
                        choices=['train', 'val', 'test'],
                        help='split type')
    parser.add_argument('--root_precomp',
                        default='/Users/mhendriksen/Desktop/repositories/SCAN/data/deep_fashion_precomp',
                        help='path to folder where to save precomputed embeds')
    args = parser.parse_args()
    main(args)