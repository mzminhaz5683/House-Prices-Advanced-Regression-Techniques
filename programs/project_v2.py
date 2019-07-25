import numpy as np # -----------------------linear algebra
import pandas as pd # ----------------------data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # ----------data manipulation
import seaborn as sns # --------------------data presentation
from scipy.stats import skew  # ------------for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import warnings
warnings.filterwarnings('ignore')

#Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

# import local file
from programs import checker_v2
checker_v2.path = path = "../output/"
######################################### START ######################################################

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_ID = train['Id']
test_ID = test['Id']

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# Deleting outliers
#train = train[train.GrLivArea < 4500]
#train.reset_index(drop=True, inplace=True)
#train["SalePrice"] = np.log1p(train["SalePrice"])
###########################################  Heat Map  ##################################################
'''
# Numerical values correlation matrix, to locate dependencies between different variables.
# Complete numerical correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(20, 13))
sns.heatmap(corrmat, vmax=1, square=True)
plt.show()

# Partial numerical correlation matrix (salePrice)
corr_num = 15 #number of variables for heatmap
cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index
corr_mat_sales = np.corrcoef(train[cols_corr].values.T)
f, ax = plt.subplots(figsize=(15, 11))
hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
plt.show()
'''
###################################### 1. Data Handling #######################################

y_train = train.SalePrice.reset_index(drop=True)
df_train = train.drop(['SalePrice'], axis = 1)
df_test = test
all_data = pd.concat([df_train, df_test]).reset_index(drop=True)
dtypes = all_data.dtypes
all_data.to_csv(path+'all_data.csv')
print('All data shape : ', all_data.shape)

######################################## missing data ###########################################
checker_v2.missing_data(all_data, 0)
#..................................... Special Case (relation) ...................................

all_data.loc[954, 'KitchenAbvGr'] = all_data['KitchenAbvGr'].mode()[0]
all_data.loc[2587, 'KitchenAbvGr'] = all_data['KitchenAbvGr'].mode()[0]
all_data.loc[2859, 'KitchenAbvGr'] = all_data['KitchenAbvGr'].mode()[0]

all_data.loc[688, 'MasVnrArea'] = all_data['MasVnrArea'].mean()
all_data.loc[1241, 'MasVnrArea'] = all_data['MasVnrArea'].mean()
all_data.loc[2319, 'MasVnrArea'] = all_data['MasVnrArea'].mean()
all_data.loc[2610, 'MasVnrType'] = all_data['MasVnrType'].mode()[0]

all_data.loc[873, 'MiscVal'] = all_data['MiscVal'].mean()
all_data.loc[1200, 'MiscVal'] = all_data['MiscVal'].mean()
all_data.loc[2431, 'MiscVal'] = all_data['MiscVal'].mean()

all_data.loc[2418, 'PoolQC'] = 'Fa'
all_data.loc[2501, 'PoolQC'] = 'Gd'
all_data.loc[2597, 'PoolQC'] = 'Fa'

all_data.loc[2124, 'GarageYrBlt'] = all_data['GarageYrBlt'].median()
all_data.loc[2574, 'GarageYrBlt'] = all_data['GarageYrBlt'].median()

all_data.loc[2124, 'GarageFinish'] = all_data['GarageFinish'].mode()[0]
all_data.loc[2574, 'GarageFinish'] = all_data['GarageFinish'].mode()[0]

all_data.loc[2574, 'GarageCars'] = all_data['GarageCars'].median()

all_data.loc[2124, 'GarageArea'] = all_data['GarageArea'].median()
all_data.loc[2574, 'GarageArea'] = all_data['GarageArea'].median()

all_data.loc[2124, 'GarageQual'] = all_data['GarageQual'].mode()[0]
all_data.loc[2574, 'GarageQual'] = all_data['GarageQual'].mode()[0]

all_data.loc[2124, 'GarageCond'] = all_data['GarageCond'].mode()[0]
all_data.loc[2574, 'GarageCond'] = all_data['GarageCond'].mode()[0]

#group = ['MasVnrArea', 'MasVnrType']
#relation = all_data[all_data['MasVnrType'].isnull()]
#checker_v2.partial(group, relation)

bsmt = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1',
        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']

#b_data = all_data[bsmt]
#b_data_null = b_data[b_data.isnull().any(axis=1)]

#group = b_data_null
#relation = b_data_null[(b_data_null.isnull()).sum(axis=1) < 5]
#checker_v2.partial(group, relation)

all_data.loc[332, 'BsmtFinType2'] = 'ALQ'
all_data.loc[947, 'BsmtExposure'] = 'No' 
all_data.loc[1485, 'BsmtExposure'] = 'No'
all_data.loc[2038, 'BsmtCond'] = 'TA'
all_data.loc[2183, 'BsmtCond'] = 'TA'
all_data.loc[2215, 'BsmtQual'] = 'Po'
all_data.loc[2216, 'BsmtQual'] = 'Fa'
all_data.loc[2346, 'BsmtExposure'] = 'No'
all_data.loc[2522, 'BsmtCond'] = 'Gd'

# fixing wrong data
all_data.loc[2592, 'GarageYrBlt'] = 2007

#group = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
#relation = all_data[all_data['YrSold'] > 2019]

#group = ['MoSold']
#relation = all_data[all_data['MoSold'] > 12]
#checker_v2.partial(group, relation)

#.................................. Multi-level .................................................
# (categorical) converting numerical variables that are actually categorical
# 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd'
cols = ['MSSubClass', 'YrSold', 'MoSold']
for var in cols:
    all_data[var] = all_data[var].astype(str)

# 'NA' means Special value
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")
all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA")

#'NA' means most frequest value
#'Utilities', 'ExterQual','Street', # categorical(numerical)
#'ExterCond','HeatingQC','LandSlope', 'LotShape','LandContour', 'LotConfig', 'BldgType',
#'RoofStyle', 'Foundation','SaleCondition'
common_vars = [ 'Exterior1st', 'Exterior2nd', 'SaleType']
for var in common_vars:
    all_data[var] = all_data[var].fillna(all_data[var].mode()[0])

# categorical 'NA' means 'None'
#'MiscFeature', 'MasVnrType', 'GarageType', 'Fence'
common_vars = ['PoolQC',
               'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
               'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in common_vars:
    all_data[col] = all_data[col].fillna('None')

#numerical 'NA' means 0
common_vars = ['GarageYrBlt', 'GarageArea', 'GarageCars']
for col in common_vars:
    all_data[col] = all_data[col].fillna(0)

# 'NA'means most or recent common according to base on other special groups
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

######################################## data classifying ############################################

# Collecting all object type feature
objects = []
for i in all_data.columns:
    if all_data[i].dtype == object:
        objects.append(i)

all_data.update(all_data[objects].fillna('None'))

# Collectting all numeric type feature
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in all_data.columns:
    if all_data[i].dtype in numeric_dtypes:
        numerics.append(i)
all_data.update(all_data[numerics].fillna(0))

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in all_data.columns:
    if all_data[i].dtype in numeric_dtypes:
        numerics2.append(i)

# checking skew
skew_all_data = all_data[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_all_data[skew_all_data > 0.5]
skew_index = high_skew.index

# Applying boxcox_normmax on skews > 0.5
for i in skew_index:
    all_data[i] = boxcox1p(all_data[i], boxcox_normmax(all_data[i] + 1))

checker_v2.missing_data(all_data, 0)

################################# subtracting and adding new feature ################################
'''
#checking the classes each categorical feature use
objects = []
for i in all_data.columns:
    if all_data[i].dtype == object:
        objects.append(i)

var_all_data = all_data[objects].apply(lambda x: len(np.unique(x))).sort_values(ascending=False)

print('------------- count of use of each variable --------------')
var_column = ['Neighborhood', 'MSSubClass', 'Exterior2nd', 'Exterior1st', 'MoSold', 'Condition1',
              'SaleType', 'RoofMatl', 'HouseStyle', 'Condition2', 'GarageType', 'Functional',
              'BsmtFinType1', 'BsmtFinType2', 'Heating', 'RoofStyle', 'Foundation',
              'SaleCondition','BsmtQual', 'GarageQual', 'FireplaceQu', 'GarageCond', 'MSZoning',
              'LotConfig', 'YrSold', 'MiscFeature', 'Fence', 'BldgType', 'HeatingQC',
              'Electrical', 'ExterCond','BsmtExposure', 'BsmtCond', 'GarageFinish', 'MasVnrType',
              'LotShape', 'LandContour','PoolQC', 'KitchenQual','ExterQual', 'Utilities',
              'Alley', 'PavedDrive','LandSlope', 'CentralAir', 'Street']
for i in var_column:
    print(all_data[i].value_counts())
    print('\n~~~~~~~~~~~~~~~~~~~~~')
'''

# dropping the columns which have a large amount of distance between it's value used amount
all_data = all_data.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

# YrBltAndRemod
all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']

# TotalSF
all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# Total_sqr_footage
all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +
                                 all_data['1stFlrSF'] + all_data['2ndFlrSF'])

# Total_sqr_footage
all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

# Total_porch_sf
all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
                              all_data['WoodDeckSF'])

# new simplified feature
all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

'''
# get only column names and transposes(T) row into columns
cName_all_data = all_data.head(0).T
cName_all_data.to_csv('../output/column_name_all_data.csv')
all_data.to_csv('../output/all_data.csv')


df_train = all_data.iloc[:len(y_train), :]
df_test = all_data.iloc[len(df_train):, :]
df_train['SalePrice'] = y_train
df_train['SalePrice'] = y_train
'''



#################################### creating dummy & de-couple all_data #############################

final_all_data = pd.get_dummies(all_data).reset_index(drop=True) # dummy

#de-couple dummy data
final_train = final_all_data.iloc[:len(y_train), :]
final_test = final_all_data.iloc[len(final_train):, :]

outliers = [30, 88, 462, 631, 1322]
final_train = final_train.drop(final_train.index[outliers])
y_train = y_train.drop(y_train.index[outliers])

# Removes columns where the threshold of zero's is (> 99.95), means has only zero values
overfit = []
for i in final_train.columns:
    counts = final_train[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(final_train) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)
overfit.append('MSZoning_C (all)')
final_train = final_train.drop(overfit, axis=1).copy()
final_test = final_test.drop(overfit, axis=1).copy()

print('final shape (df_train, y_train, df_test): ',final_train.shape,y_train.shape,final_test.shape)


def get_train_label():
    print("y_train of get_train_label():", y_train.shape)
    return y_train

def get_test_ID():
    print("df_test_ID of get_test_ID():", test_ID.shape)
    return test_ID

def get_train_test_data():
    print('final shape of get_train_test_data(): ', final_train.shape, y_train.shape, final_test.shape)
    return final_train, final_test