import argparse
import config
from model import deep_clustering, utils
import logger

# arguments parser
parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - DEEP EMBEDDED CLUSTERING MODEL', formatter_class=argparse.RawDescriptionHelpFormatter, description="Preprocess Audio Dataset")
parser.add_argument("-c", "--config", required=True, help="config file")
parser.add_argument("-u", "--uuid", required=True, help="UUID of the preprocessed dataset to use")

if __name__ == "__main__":
    args = parser.parse_args()

    config = config.Config(path=args.config)
    logger = logger.get_logger()

    dataset_loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger)

    # deep embedded clustering model
    clustering_model = deep_clustering.ClusteringModel(dataset_loader, logger=logger, pre_train_epochs=250)
    # clustering_model.train(clustering_epochs=500, update_freq=5)