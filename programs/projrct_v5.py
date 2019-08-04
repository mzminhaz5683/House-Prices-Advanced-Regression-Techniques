# coding=utf-8
import numpy as np # -------------------linear algebra
import pandas as pd # ------------------data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # ------data manipulation
import seaborn as sns # ----------------data presentation

import warnings
warnings.filterwarnings('ignore')

#Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.1f}'.format(x))
p_description = ''

# import local file
from programs import checker_v2
checker_v2.path = path = "../output/Data_observation/"
######################################## Controller ##############################################
missing_data = 0
save_all_data = 0
value_counter = 0
hit_map = 0
column_name = 0
file_description = 1
check_outliars_numeric = 0
check_outliars_objects = 0
file_open_order = 'w'
######################################## design #(40) + string + #(*) = 100 ######################
######################################## START ###################################################

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_ID = train['Id'] # assign
test_ID = test['Id'] # assign

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

######################################## Heat Map  ################################################
if hit_map:
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
    f, ax = plt.subplots(figsize=(20, 15))
    hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 7}, yticklabels=cols_corr.values,
                     xticklabels=cols_corr.values)
    plt.show()
######################################## 1. Data Handling #########################################

y_train = train.SalePrice.reset_index(drop=True) # assign
df_train = train.drop(['SalePrice'], axis = 1) # drop
df_test = test # assign
all_data = pd.concat([df_train, df_test]).reset_index(drop=True) # concatenation
dtypes = all_data.dtypes
if save_all_data:
    all_data.to_csv(path+'all_data_prime.csv')
    print('\nAll prime data has been saved at : '+path+'all_data_prime.csv')
print('____________________________________________________________________________________')
print('all_data shape (Rows, Columns) & Columns-(ID, SalePrice): ', all_data.shape)

######################################## missing data operation ##################################
if missing_data:
    checker_v2.missing_data(all_data, 0) # checking missing data 1/0 for save as file or not

#.........................................dealing with missing data.....................................
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
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
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

'''
# checking skew
skew_all_data = all_data[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_all_data[skew_all_data > 0.5]
skew_index = high_skew.index

# Applying boxcox_normmax on skews > 0.5
for i in skew_index:
    all_data[i] = boxcox1p(all_data[i], boxcox_normmax(all_data[i] + 1))
'''
p_description += 'remove skew\n'
################################# subtracting and adding new feature ################################

# dropping too much missing data columns
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


df_train = all_data.iloc[:len(y_train), :]
df_test = all_data.iloc[len(df_train):, :]
df_train['SalePrice'] = y_train

df_train = df_train[df_train.GrLivArea < 4500]
df_train.reset_index(drop=True, inplace=True)
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
#################################### creating dummy & de-couple all_data #############################
y_train = df_train.SalePrice.reset_index(drop=True)
df_train = df_train.drop(['SalePrice'], axis = 1)

final_data = pd.concat([df_train, df_test]).reset_index(drop=True)

if save_all_data:
    final_data.to_csv(path+'all_data_final.csv')
    print('____________________________________________________________________________________')
    print('all_final_data has been saved at (Before Dummy): '+path+'all_data_final.csv')
final_all_data = pd.get_dummies(final_data).reset_index(drop=True) # dummy

#de-couple dummy data
final_train = final_all_data.iloc[:len(y_train), :]
final_test = final_all_data.iloc[len(final_train):, :]

#outliers = [30, 88, 462, 631, 1322]
#final_train = final_train.drop(final_train.index[outliers])
#y_train = y_train.drop(y_train.index[outliers])
# Removes columns where the threshold of zero's is (> 99.95), means has only zero values

overfit = []
for i in final_train.columns:
    counts = final_train[i].value_counts()
    zeros = counts.iloc[0]
    if ((zeros / len(final_train)) * 100) > 99.94:
        overfit.append(i)

overfit = list(overfit)
#overfit.append('MSZoning_C (all)')
print('overfit : ', overfit)
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
    print('Shape of get_train_test_data(): ', final_train.shape, y_train.shape, final_test.shape)
    return final_train, final_test

if file_description:
    p_description += '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n'
    file = open(path+'process_description.txt', file_open_order)
    file.write(p_description)
    print('\n____________________________________________________________________________________')
    print('process_description has been saved at : '+path+'process_description.txt')
    file.close()