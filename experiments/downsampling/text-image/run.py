import configargparse
import logging
import pickle
import random
from shutil import copyfile
import torch

import platalea.text_image as M
import platalea.dataset as D
from utils.copy_best import copy_best

# Parsing arguments
parser = configargparse.get_argument_parser('platalea')
parser.add_argument(
    '--seed', default=123, type=int,
    help='seed for sources of randomness (default: 123)')
config_args, _ = parser.parse_known_args()

# Setting general configuration
torch.manual_seed(config_args.seed)
random.seed(config_args.seed)
logging.basicConfig(level=logging.INFO)


batch_size = 32
hidden_size = 1024
dropout = 0.0

factors = [3, 9, 27, 81, 243]
lz = len(str(abs(factors[-1])))
for ds_factor in factors:
    logging.info('Loading data')
    data = dict(
        train=D.flickr8k_loader(split='train', batch_size=batch_size,
                                shuffle=True, downsampling_factor=ds_factor),
        val=D.flickr8k_loader(split='val', batch_size=batch_size,
                              shuffle=False))
    fd = D.Flickr8KData
    fd.init_vocabulary(data['train'].dataset)

    # Saving config
    pickle.dump(dict(label_encoder=fd.get_label_encoder(),
                     language='en'),
                open('config.pkl', 'wb'))

    logging.info('Building model')
    net = M.TextImage(M.get_default_config())
    run_config = dict(max_lr=2 * 1e-4, epochs=32)

    logging.info('Training')
    M.experiment(net, data, run_config)
    suffix = str(ds_factor).zfill(lz)
    res_fname = 'result_{}.json'.format(suffix)
    copyfile('result.json', res_fname)
    copy_best(res_fname, 'net_{}.best.pt'.format(ds_factor))
