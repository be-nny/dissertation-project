import argparse

import torch

import logger
import model

from utils import *
from model import utils as model_utils, models, trainer
from datetime import datetime

parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - MODEL', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-g", "--genres", help="Takes a comma-seperated string of genres to use (e.g., jazz,rock,blues,disco) - if set to 'all', all genres are used")
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-d", "--dec")

if __name__ == "__main__":
    args = parser.parse_args()
    config = config.Config(path=args.config)
    logger = logger.get_logger()

    if args.info:
        show_info(logger, config)

    with model_utils.ReceiptReader(filename=os.path.join(config.OUTPUT_PATH, f'{args.uuid}/receipt.json')) as receipt:
        signal_processor = receipt.signal_processor
        segment_duration = receipt.seg_dur

    folder = f"{signal_processor}_{args.uuid}_{args.genres}"
    root = f"{config.OUTPUT_PATH}/dec/{folder}"
    if not os.path.exists(root):
        os.makedirs(root)

    if args.train:
        _, genre_filter = get_genre_filter(args.genres)
        loader = model_utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=model.BATCH_SIZE)
        batch_loader = loader.load(split_type="train", normalise=True, genre_filter=genre_filter, flatten=False)
        str_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(root, f"conv_1d_autoencoder_{str_time}.pt")

        conv_ae = models.Conv1DAutoencoder(n_layers=[256, 64, 32, 16, 8], latent_dim=2, input_shape=loader.input_shape)
        trainer.train_autoencoder(epochs=1000, autoencoder=conv_ae, batch_loader=batch_loader, logger=logger, path=path, batch_size=64)

    if args.dec:
        model_path = args.dec
        _, genre_filter = get_genre_filter(args.genres)
        loader = model_utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=model.BATCH_SIZE)
        batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter, flatten=False)

        conv_ae = models.Conv1DAutoencoder(n_layers=[256, 64, 32, 16, 8], latent_dim=2, input_shape=loader.input_shape)
        conv_ae.load_state_dict(torch.load(model_path, weights_only=True))

        str_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(root, f"dec_{str_time}.pt")
        dec = models.DEC(ae=conv_ae, n_clusters=10, latent_dims=2)
        trainer.train_dec(epochs=1000, dec=dec, batch_loader=batch_loader, logger=logger, path=path)

