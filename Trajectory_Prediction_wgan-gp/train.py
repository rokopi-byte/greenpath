"""Train script.
Usage:
  train.py <hparams> <dataset_root> [--cuda=<id>]
  train.py -h | --help

Options:
  -h --help     Show this screen.
  --cuda=<id>   id of the cuda device [default: 0].
"""

import torch
import numpy as np
import json
from docopt import docopt
from os.path import join
from irl_dcb.config import JsonConfig
from dataset import process_data
from irl_dcb.builder import build
from irl_dcb.trainer import Trainer
torch.manual_seed(42620)
np.random.seed(42620)

if __name__ == '__main__':
    args = docopt(__doc__)
    device = torch.device('cuda:{}'.format(args['--cuda']) if torch.cuda.is_available() else "cpu")
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    hparams = JsonConfig(hparams)

    # dir of pre-computed beliefs
    DCB_dir_HR = join(dataset_root, 'DCBs/HR/')
    DCB_dir_LR = join(dataset_root, 'DCBs/LR/')

    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()
    print("Training, version 11...")
    # load ground-truth human scanpaths
    with open(join(dataset_root,
                   'dataset_train.json')) as json_file:
        human_scanpaths_train = json.load(json_file)
    with open(join(dataset_root,
                   'dataset_valid.json')) as json_file:
        human_scanpaths_valid = json.load(json_file)

    # exclude incorrect scanpaths
    if hparams.Train.exclude_wrong_trials:
        human_scanpaths_train = list(
            filter(lambda x: x['correct'] == 1, human_scanpaths_train))
        human_scanpaths_valid = list(
            filter(lambda x: x['correct'] == 1, human_scanpaths_valid))

    with open(join(dataset_root,
                   'first_fixations.json')) as json_file:
        fixations = json.load(json_file)

    with open(join(dataset_root,
                   'not_walkable.json')) as json_file:
        walls = json.load(json_file)

    # process fixation data
    dataset = process_data(human_scanpaths_train, human_scanpaths_valid,
                           DCB_dir_HR, DCB_dir_LR, fixations, walls, bbox_annos, hparams)
    built = None
    if hparams.Train.load_checkpoints:
        built = build(hparams, True, device, dataset['catIds'], dataset['first_fixations'], "./trained_models")
    else:
        built = build(hparams, True, device, dataset['catIds'], dataset['first_fixations'])
    trainer = Trainer(**built, dataset=dataset, device=device, hparams=hparams)
    trainer.train()
