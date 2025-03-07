# Music Analysis with Machine Learning

## The `config.yml` File
This file contains any configuration settings for preprocessing and training a model.
```yml
dataset: "/path/to/dataset"
output: "/output/directory"
preprocessor_config:
  target_length: target_length_of_all_songs_int
  segment_duration: the_snippet_length_of_each_song_int
  train_split: float_val_between_0_1
```

## Dataset Directory
The input dataset should be structure in the following way. A root directory containing a list of subdirectories named as the genre name with a series of `mp3` or `wav` files in them.
This project primarily uses the `GTZAN` dataset, which can be downloaded [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

```
dataset/
├─ genre_1/
│  ├─ song_1.mp3
│  ├─ song_2.mp3
├─ genre_2/
├─ genre_.../
├─ genre_n/
```

## Preprocessing

## Model
Running the following command will use all the genres, `-g all`, in the preprocessed dataset, `-u UUID`, with 10 clusters to produce a gaussian mixture model plot. The plot contains 3 ellipses signifying the 1st, 2nd, and 3rd standard deviations explaining the likelihood of a point belonging to a particular cluster. 
```pycon
python model.py -c config.yml -u UUID -g all -n 10
```
![gaussian_plot.png](examples/gaussian_plot.png)

The treemap shows which genres belong in which cluster.
![tree_map.png](examples/tree_map.png)