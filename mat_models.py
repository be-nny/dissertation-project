import mat_config
from mat_logger import mat_logger
from mat_models import pca_model, utils


config = mat_config.Config(path="config.yml")
logger = mat_logger.get_logger()

dataset_loader = utils.Loader(out=config.OUTPUT_PATH, uuid="6a3a52")

pca = pca_model.PCAModel(
    out=config.OUTPUT_PATH,
    uuid="6a3a52",
    n_components=6,
    loader=dataset_loader,
    logger=logger
)

pca.create()
