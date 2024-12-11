from mat_logger import mat_logger
from mat_models import pca_model, utils
from jsonschema import validate

import yaml

with open("config.yml", 'r') as f:
    yml_data = yaml.load(f, Loader=yaml.FullLoader)

# validating schema
with open("schema.yml", 'r') as schema:
    validate(yml_data, yaml.load(schema, Loader=yaml.FullLoader))

dataset_path = yml_data["dataset"]
output_path = yml_data["output"]
target_length = yml_data["preprocessor_config"]["target_length"]
segment_duration = yml_data["preprocessor_config"]["segment_duration"]
train_split = yml_data["preprocessor_config"]["train_split"]

if train_split > 1:
    raise ValueError("'train_split' must be <= 1")

logger = mat_logger.get_logger()

dataset_loader = utils.Loader(out=output_path, uuid="5c431b")
pca = pca_model.PCAModel(
    out=output_path,
    uuid="5c431b",
    n_components=6,
    loader=dataset_loader,
    logger=logger
)

pca.create()
pca.plot()