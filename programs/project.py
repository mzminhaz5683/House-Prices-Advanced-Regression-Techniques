import pandas as pd # data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # data manipulation
import seaborn as sns # data presentation
import numpy as np # linear algebra
from scipy.stats import norm #for some statistics
from scipy import stats  # scientific notation handler
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
from programs import checker # import local file

#.....Importing & Checking Inputs.")
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print("df_train set size:", df_train.shape) #1460 samples
print("df_test set size:", df_test.shape) # 1459 df_test cases

#.....Dropping 'Id' column since it's not a necessary item on prediction
df_train_ID = df_train['Id']
df_test_ID = df_test['Id']
df_train.drop(['Id'], axis=1, inplace=True)
df_test.drop(['Id'], axis=1, inplace=True)

#.....work with outliers of GrLivArea
#checker.scatter(df_train, 'GrLivArea')
#.....outliers : GrlivArea > 4500
df_train = df_train[df_train.GrLivArea < 4500]
df_train.reset_index(drop=True, inplace=True)
#.....after removing outliers from GrLivArea
#checker.scatter(df_train, 'GrLivArea')

#.....work with outliers of TotalBsmtSF
#checker.scatter(df_train, 'TotalBsmtSF')