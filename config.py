import os

import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError

class Config:
    def __init__(self, path):
        self.path = path

        # reading config file
        try:
            with open(self.path, 'r') as f:
                yml_data = yaml.load(f, Loader=yaml.FullLoader)
        except (yaml.YAMLError, FileNotFoundError) as e:
            raise IOError(f"Error reading config file '{self.path}': {e}")

        # validating schema
        try:
            with open("schema.yml", 'r') as schema:
                validate(yml_data, yaml.load(schema, Loader=yaml.FullLoader))
        except (yaml.YAMLError, FileNotFoundError) as e:
            raise IOError(f"Error reading schema file '{self.path}': {e}")
        except ValidationError as e:
            raise ValueError(f"Schema validation error: {e}")

        if yml_data["preprocessor_config"]["train_split"] > 1:
            raise ValueError("'train_split' must be <= 1")

        self.DATASET_PATH = yml_data["dataset"]
        self.OUTPUT_PATH = yml_data["output"]
        self.TARGET_LENGTH = yml_data["preprocessor_config"]["target_length"]
        self.SEGMENT_DURATION = yml_data["preprocessor_config"]["segment_duration"]
        self.TRAIN_SPLIT = yml_data["preprocessor_config"]["train_split"]

