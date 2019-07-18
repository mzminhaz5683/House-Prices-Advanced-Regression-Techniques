import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O handler(e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt # data presentation

from datetime import datetime # time handler
from scipy import stats # scientific notation handler
from scipy.stats import norm, skew #for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline

# Input data files are available in the "./input/" directory.

import os
print(os.listdir("./input"))

import warnings
warnings.filterwarnings('ignore')
