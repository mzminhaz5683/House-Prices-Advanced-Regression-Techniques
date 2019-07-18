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

#.....Importing & Checking Inputs.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print("df_train set size:", df_train.shape) #1460 samples
print("df_test set size:", df_test.shape) # 1459 df_test cases

#.....Dropping 'Id' column since it's not a necessary item on prediction
df_train_ID = df_train['Id']
df_test_ID = df_test['Id']
df_train.drop(['Id'], axis=1, inplace=True)
df_test.drop(['Id'], axis=1, inplace=True)


# 1. Missing Data Handling
# {
#.....missing data observing
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data)

#.....dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index) #dropping the observation
#print(df_train.isnull().sum().max()) #just checking that there's no missing data missing...
# }


# 2. Out-liars Handling
# {
# 2a.....numerical analyzing
#   [
#checker.numerical_relationship(df_train, 'GrLivArea')
df_train = df_train[df_train.GrLivArea < 4500] # outliers : GrlivArea > 4500
df_train.reset_index(drop=True, inplace=True) # removing outliers from GrLivArea
#checker.numerical_relationship(df_train, 'GrLivArea')

#checker.numerical_relationship(df_train, 'TotalBsmtSF')
#   ]

# 2b.....categorical analyzing
#   [
#checker.categorical_relationship(df_train, 'OverallQual')
#checker.categorical_relationship(df_train, 'YearBuilt')
#   ]
# }