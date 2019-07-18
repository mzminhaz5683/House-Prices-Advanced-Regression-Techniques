import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O handler(e.g. pd.read_csv)
import seaborn as sns # # data presentation
import matplotlib.pyplot as plt # data manipulation

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
#.....Input data files are available in the "../input/" directory.
import os
#print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')
#......importing local files
from programs import checker
#.....Importing & Checking Inputs
#print("imported data..")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#print("Train set size:", train.shape) #1460 samples
#print("Test set size:", test.shape) # 1459 test cases

#.....test inputs : for pycharm use print() to project even any direct printing command of python
#print(train.head())
#print(test.head())

#.....Dropping 'Id' column since it's not a necessary item on prediction
train_ID = train['Id']
test_ID = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

checker.general_distribution(train, 'SalePrice')
#.....Deleting the more visibly obvious outliers
#.....4500 exceeds the central tendency of of the houses in that price point and all the houses in the dataset
#.....will deal with more subtle outlets later.
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)
#.....it will update the train file according to normalization
checker.normalized_distribution(train, 'SalePrice')
checker.general_distribution(train, 'SalePrice')