# COMP9444 Project Requirements

import os, sys, json, glob, random, time
import numpy as np
from tabulate import tabulate

import imageio as io
from PIL import Image
from IPython.display import display

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

import torch, torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from shapely.geometry import Polygon
import geopandas as gpd

IMAGE_ENDS = '_leftImg8bit.png'
LABEL_ENDS = '_gtBboxCityPersons.json'
PATH = os.path.realpath(__file__)
DIR = os.path.dirname(PATH)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CHANGE PATHS AS NECESSARY
IMAGE_PATH = '/home/nicholas/Datasets/CityPersons/Cityshapes_images/leftImg8bit/'
LABEL_PATH = '/home/nicholas/Datasets/CityPersons/CityPersons_labels/gtBboxCityPersons/'


