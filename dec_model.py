import argparse
import config
from model import deep_clustering, utils
import logger

# arguments parser
parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - DEEP EMBEDDED CLUSTERING MODEL', formatter_class=argparse.RawDescriptionHelpFormatter, description="Preprocess Audio Dataset")
parser.add_argument("-c", "--config", required=True, help="config file")
parser.add_argument("-l", "--layers", nargs="+", required=True, help="hidden layer sizes", type=int)
parser.add_argument("-u", "--uuid", required=True, help="UUID of the preprocessed dataset to use")

config = config.Config(path="config.yml")
logger = logger.get_logger()

# loading all data (combining test and train splits together)
dataset_loader = utils.Loader(out=config.OUTPUT_PATH, uuid="aa2884", logger=logger)

# deep embedded clustering model
clustering_model = deep_clustering.ClusteringModel(dataset_loader, logger=logger, hidden_layers=[512, 512, 1024, 128, 5], pre_train_epochs=50)
clustering_model.train(clustering_epochs=500, update_freq=3)