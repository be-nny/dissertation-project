import matplotlib
import numpy as np
import pandas as pd
import pypalettes
import squarify
from kneed import KneeLocator

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

matplotlib.use('TkAgg')

colour_map_name = "hat"
CMAP = pypalettes.load_cmap(colour_map_name)