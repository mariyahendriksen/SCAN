from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import logging
import argparse
from collections import defaultdict
import pickle

from layers_resnest import Layers_resnest, transform_mlf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def get_image_features_dict(model, root, file_paths):
    features_dict = defaultdict()
    for file_path in tqdm(file_paths):
        if file_path not in features_dict:
            path = os.path.join(root, file_path)
            img = Image.open(path).convert('RGB')
            try:
                img_transformed = transform_mlf(img).unsqueeze(0).to(device)
            except:
                print('Problem with ', file_path)
            feature = model.forward1(img_transformed).to('cpu').squeeze().detach().numpy()
            features_dict[file_path] = feature

    return features_dict

def get_image_features(model, root, file_paths):
    features = defaultdict()
    for file_path in tqdm(file_paths):
        path = os.path.join(root, file_path)
        img = Image.open(path).convert('RGB')
        try:
            img_transformed = transform_mlf(img).unsqueeze(0).to(device)
        except:
            print('Problem with ', file_path)
        feature = model.forward1(img_transformed).to('cpu').squeeze().detach().numpy()
        features.append(feature)
    features = np.stack(features, axis=0)
    return features


def main(args):
    dataset_type = args.dataset_type
    root_precomp = f'/ivi/ilps/personal/mbiriuk/repro/data/{dataset_type}_mlf_precomp'
    logging.info(f"""
    dataset type - {dataset_type}
    root_precomp - {root_precomp}
    """)

    ############################################
    # 1. loads:
    ############################################
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
        df = pd.read_csv(
            df_file,
            dtype={
                'image_name': str,
                'eval_status': str,
                'product_type': str
            },
            index_col=0
        )
    elif dataset_type == 'f30k':
        df_file = '/ivi/ilps/personal/mbiriuk/data/data/flickr30k_images/dataset_flickr30k.csv'
        root_raw = '/ivi/ilps/personal/mbiriuk/data/data/flickr30k_images/flickr30k_images'
        df = pd.read_csv(
            df_file,
            dtype={
                'image_name': str,
                'eval_status': str,
                'image_caption': str
            },
            index_col=0
        )
    elif dataset_type == 'cub':
        df_file = '/ivi/ilps/personal/mbiriuk/CUB_200_2011/cub_captions.csv'
        root_raw = '/ivi/ilps/personal/mbiriuk/CUB_200_2011/images'
        df = pd.read_csv(
            df_file,
            dtype={
                'image_file': str,
                'eval_status': str,
                'caption': str
            },
            index_col=0
        )
    else:
        raise NotImplementedError
    logging.info('Loaded df')
    file_paths = df['image_file'].unique().tolist()
    logging.info(f'Got image_names from {dataset_type}, unique files - {len(file_paths)}')

    ############################################
    # 2. embed images
    ############################################
    features_dict = get_image_features_dict(net, root_raw, file_paths)
    logging.info('Completed image transformations, final dict size: ', len(features_dict))

    ############################################
    # 3. save image representations as dict
    ############################################
    os.makedirs(root_precomp, exist_ok=True)
    target_file = os.path.join(root_precomp, 'ims_dict.pkl')
    with open(target_file, 'wb+') as f:
        pickle.dump(features_dict, f)
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
    args = parser.parse_args()
    main(args)