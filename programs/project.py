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


###################################### 1. Missing Data Handling #########################################
# {
#.....missing data observing
ntrain = df_train.shape[0]
y_train = df_train.SalePrice.values

df_train.drop(['SalePrice'], axis = 1, inplace = True)
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

total = all_data.isnull().sum().sort_values(ascending=False)
percent = ((all_data.isnull().sum()/all_data.isnull().count()) * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data)



#.....dealing with missing data
### imputing missing values
# PoolQC --> NA means missing houses have no Pool in general so "None"
all_data['PoolQC'] = all_data['PoolQC'].fillna("None")

# MiscFeature --> NA means no misc. features so "No"
all_data['MiscFeature'] = all_data['MiscFeature'].fillna("None")

# Alley :  NA means "no alley access"
all_data['Alley'] = all_data['Alley'].fillna("None")

# Fence: NA means "no fence"
all_data['Fence'] = all_data['Fence'].fillna("None")

# FireplaceQu: NA means "no fireplace"
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna("None")


# GarageType, GarageFinish, GarageQual and GarageCond: NA means "None"
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna("None")

# GarageYrBlt, GarageArea and GarageCars : NA means o
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath:
# "NA" means 0 for no basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

# 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
# categorical meaning NA means 'None'
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

# masonry veneer: 0 for area and None for category
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# Utilises: won't help in predictive modelling
all_data.drop(['Utilities'], axis = 1, inplace = True) #?????????????????????????????????????????


# Functional: NA means typical
all_data['Functional'] = all_data['Functional'].fillna('Typ')


# set the most common string
# MSZoning: NA replace most common value of the list "RL"
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# Electrical: NA means SBrkr
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


#SaleType: NA means WD
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# KitchenQual
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# Exterior1st and Exterior2nd: NA means most common string
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# most important
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# converting numerical variables that are actually categorical
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# creating a set of all categorical variables
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features
from sklearn.preprocessing import LabelEncoder #import labelEncoder to process data
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

# separate all_data into df_train & df_test
df_train = all_data[:ntrain]
df_test = all_data[ntrain:]
df_train['SalePrice'] = y_train #adding 'SalePrice' column into df_train
#print(df_train.isnull().sum().max()) #just checking that there's no missing data missing...
# }

#cName_train = df_train.head(0).T # get only column names and transposes(T) row into columns
#cName_train.to_csv('../output/column_name_ms_train.csv') # save accepted column names after handling missing data
#df_train.to_csv('../output/ms_train.csv')
print("df_train set size after handling missing data (remove : id & utilities):", df_train.shape)
print("df_test set size after handling missing data(remove : id & utilities):", df_test.shape)


##################################### 2. Out-liars Handling #############################################
# {
# 2a.....numerical analyzing
#       [
#checker.numerical_relationship(df_train, 'GrLivArea')
df_train = df_train[df_train.GrLivArea < 4500] # outliers : GrlivArea > 4500
df_train.reset_index(drop=True, inplace=True) # removing outliers from GrLivArea
#checker.numerical_relationship(df_train, 'GrLivArea')

#checker.numerical_relationship(df_train, 'TotalBsmtSF')
#       ]


# 2b.....categorical analyzing
#       [
#checker.categorical_relationship(df_train, 'OverallQual')
#checker.categorical_relationship(df_train, 'YearBuilt')
#       ]
# }



###################################### 3. Normalization handling #########################################
# {
#checker.general_distribution(df_train, 'SalePrice')
#plt.show()
checker.normalized_distribution(df_train, 'SalePrice')
#plt.show()


#checker.general_distribution(df_train, 'GrLivArea')
#plt.show()
checker.normalized_distribution(df_train, 'GrLivArea')
#plt.show()


#checker.general_distribution(df_train, 'TotalBsmtSF')
#plt.show()
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
checker.general_distribution(df_train, 'TotalBsmtSF')
#plt.show()
# }


################################### 3. Homoscedasticity handling #########################################
# {
# }

############################ 4. Converting categorical variable into dummy ###############################
#all_data = pd.get_dummies(all_data) #?????????????????????????????????????????
