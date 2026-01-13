#!/usr/bin/env python3
"""
Train larger EfficientNet variants (B1, B2, B3) on HAM10000 to achieve 98% accuracy.
"""

import os
import sys
import logging
import time
from datetime import datetime
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import 