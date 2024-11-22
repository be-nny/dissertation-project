# Music Analysis with Machine Learning

## `config.env` file 
A `config.env` file must be created before running the tool. The config file must include:

```dotenv
DATASET_PATH="path/to/dataset"
FIGURES_PATH="path/to/figure/output/dir"
PREPROCESSED_PATH="path/to/preprocessed/output/dir"
```

## Preprocessing
Run the following to start the processing phase
```pycon
python main.py -c config.env -p
```

## Generating Example Figures
Run the following to generate some preprocessing examples
```pycon
python main.py -c config.env -p
```