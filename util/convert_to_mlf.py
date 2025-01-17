from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import logging
import argparse
import pickle

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


def get_image_features(
        # model,
        # root,
        file_paths,
        features_dict):
    features = []
    for file_path in tqdm(file_paths):
        feature = features_dict[file_path]
        # path = os.path.join(root, file_path)
        # img = Image.open(path).convert('RGB')
        # try:
        #     img_transformed = transform_mlf(img).unsqueeze(0).to(device)
        # except:
        #     print('Problem with ', file_path)
        try:
            feature = features_dict[file_path]
        except:
            print('Problem with ', file_path)
        features.append(feature)

    features = np.stack(features, axis=0)
    return features


def main(args):
    dataset_type = args.dataset_type
    split = args.split
    root_precomp = f'/ivi/ilps/personal/mbiriuk/repro/data/{dataset_type}_mlf_precomp'
    logging.info(f"""
    dataset type - {dataset_type}
    split - {split}
    root_precomp - {root_precomp}
    """)

    # 1. loads:
    # layers model
    net = Layers_resnest(img_dim=2048, trained_dresses=False, checkpoint_path=None)
    net.to(device)
    net.eval()
    logging.info('Loaded layers resnest')
    # df
    if dataset_type == 'deep_fashion':
        df_file = '/ivi/ilps/personal/mbiriuk/data/data/deep_fashion/category-and-attribute-prediction/eval_partition' \
                  '.csv '
        root_raw = '/ivi/ilps/personal/mbiriuk/data/data/deep_fashion/category-and-attribute-prediction'
        dict_path = ''
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
    elif dataset_type == 'f30k':
        df_file = '/ivi/ilps/personal/mbiriuk/data/data/flickr30k_images/dataset_flickr30k.csv'
        root_raw = '/ivi/ilps/personal/mbiriuk/data/data/flickr30k_images/flickr30k_images'
        dict_path = ''
        df = pd.read_csv(
            df_file,
            dtype={
                'image_name': str,
                'eval_status': str,
                'image_caption': str
            },
            index_col=0
        )
        logging.info('Loaded df')
        df_subset = df[df['eval_status'] == split]
        file_paths = df_subset['image_name'].tolist()
    elif dataset_type == 'cub':
        df_file = '/ivi/ilps/personal/mbiriuk/CUB_200_2011/cub_captions.csv'
        root_raw = '/ivi/ilps/personal/mbiriuk/CUB_200_2011/images'
        dict_path = '/ivi/ilps/personal/mbiriuk/repro/data/cub_mlf_precomp/ims_dict.pkl'
        df = pd.read_csv(
            df_file,
            dtype={
                'image_file': str,
                'eval_status': str,
                'caption': str
            },
            index_col=0
        )
        with open(dict_path, 'rb') as f:
            features_dict = pickle.load(f)
        df_subset = df[df['eval_status'] == split]
        file_paths = df_subset['image_file'].tolist()
        logging.info('Loaded df and images dict, got file paths')
    else:
        raise NotImplementedError
    logging.info(f'Got image_names from {dataset_type} split: {split}')
    # 2. embed images
    features = get_image_features(
        # net, root_raw,
        file_paths, features_dict)
    logging.info(f'Completed image transformations, final images shape: {features.shape}')

    # 3. save image representations
    os.makedirs(root_precomp, exist_ok=True)
    target_file = os.path.join(root_precomp, f'{split}_ims.npy')
    with open(target_file, 'wb+') as f:
        np.save(f, features)
    logging.info(f"Saved image file to {target_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type',
                        default='cub',
                        choices=[
                            'f30k',
                            'deep_fashion',
                            'cub'
                        ],
                        help='f30k or deep fashion dataset')
    parser.add_argument('--split',
                        default='train',
                        choices=['train', 'test', 'dev'],
                        help='split type')
    args = parser.parse_args()
    main(args)
