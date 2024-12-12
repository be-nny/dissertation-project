import mat_config
from mat_logger import mat_logger
from mat_models import dimensionality_reduction as dr, utils


config = mat_config.Config(path="config.yml")
logger = mat_logger.get_logger()

dataset_loader = utils.Loader(out=config.OUTPUT_PATH, uuid="9378f8")

pca = dr.PCAModel(
    out=config.OUTPUT_PATH,
    uuid="9378f8",
    n_components=6,
    loader=dataset_loader,
    logger=logger
)

pca.create()
