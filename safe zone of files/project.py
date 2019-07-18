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
import os
#print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')
from programs import checker


#.....Importing & Checking Inputs.")
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#print("df_train set size:", df_train.shape) #1460 samples
#print("df_test set size:", df_test.shape) # 1459 df_test cases


#.....Dropping 'Id' column since it's not a necessary item on prediction
df_train_ID = df_train['Id']
df_test_ID = df_test['Id']
df_train.drop(['Id'], axis=1, inplace=True)
df_test.drop(['Id'], axis=1, inplace=True)

#.....check distribution of Sale Price from df_train, as it's in only there.
checker.general_distribution(df_train, 'SalePrice')
#.....Deleting the more visibly obvious outliers
#.....4500 exceeds the central tendency of of the houses in that price point and all the houses in the dataset
#.....will deal with more subtle outlets later.
df_train = df_train[df_train.GrLivArea < 4500]
df_train.reset_index(drop=True, inplace=True)
#.....it will update the df_train file according to normalization
checker.normalized_distribution(df_train, 'SalePrice')
checker.general_distribution(df_train, 'SalePrice')