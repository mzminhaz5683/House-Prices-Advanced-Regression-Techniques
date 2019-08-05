# coding=utf-8
import numpy as np # -------------------linear algebra
import pandas as pd # ------------------data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # ------data manipulation
import seaborn as sns # ----------------data presentation

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import skew

import warnings
warnings.filterwarnings('ignore')

#Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.1f}'.format(x))
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Raw : 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
p_description = ''

# import local file
from programs import checker_v2
checker_v2.path = path = "../output/Data_observation/"
######################################## Controller ##############################################
missing_data = 0
save_all_data = 0
column_value_use_counter = 0
hit_map = 0
save_column_name = 0
file_description = 1

check_outliars_numeric = 0
check_outliars_objects = 0

file_open_order = 'w'
single_level_Data_Handling = 0
o2n_converter = 0
multi_level_modified = 1
normalization = 1
######################################## design #(40) + string + #(*) = 100 ######################




######################################## START ###################################################
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Raw : 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_ID = train['Id'] # assign
test_ID = test['Id'] # assign

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)



######################################## Heat Map : 1 ################################################
if hit_map:
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

######################################## 2. Out-liars Handling ##################################
#.......................................2a numerical analyzing...................................

p_description += '---------------- Numerical_outliars : Top ---------------------\n'
if check_outliars_numeric:
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in train.columns:
        if train[i].dtype in numeric_dtypes:
            numerics.append(i)

    if save_column_name:
        # get only column names and transposes(T) row into columns
        numeric_data = train[numerics]
        cName_n_data = numeric_data.head(0).T
        cName_n_data.to_csv(path + 'numeric_save_column_names.csv')
        print('Numeric columns names saved at :' + path + 'numeric_save_column_names.csv')

    if 0:
        for i in numerics:
            checker_v2.numerical_relationship(train, i)
    else:
        numerics_outliars = ['TotalBsmtSF', '1stFlrSF', 'GrLivArea']
        for i in numerics_outliars:
            checker_v2.numerical_relationship(train, i)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

drop_index = train[(train['TotalBsmtSF'] > 6000) & (train['SalePrice']<200000)].index
train = train.drop(drop_index)

p_description += "drop_index = train[(train['TotalBsmtSF'] > 6000)" \
                 " & (train['SalePrice']<200000)].index\n"
#________________________________________________________________________________________________

drop_index = train[(train['1stFlrSF'] > 4500) & (train['SalePrice']<200000)].index
train = train.drop(drop_index)

p_description += "drop_index = train[(train['1stFlrSF'] > 4500)" \
                 " & (train['SalePrice']<200000)].index\n"
# ________________________________________________________________________________________________

drop_index = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index
train = train.drop(drop_index)

p_description += "drop_index = train[(train['GrLivArea'] > 4000) " \
                 "& (train['SalePrice'] < 300000)].index\n"
# ________________________________________________________________________________________________
p_description += '-----------------------------------------------------------------\n'
#________________________________________________________________________________________________




#.......................................2b categorical analyzing.................................
if check_outliars_objects:
    objects = []
    for i in train.columns:
        if train[i].dtype == object:
            objects.append(i)

    if save_column_name:
        # get only column names and transposes(T) row into columns
        object_data = train[objects]
        cName_n_data = object_data.head(0).T
        cName_n_data.to_csv(path + 'object_save_column_names.csv')
        print('Object columns names saved at :' + path + 'Object_save_column_names.csv')

    if 1:
        for i in objects:
            checker_v2.categorical_relationship(train, i)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Raw : 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

#checker_v2.general_distribution(df_train, 'SalePrice')
#plt.show()
#checker_v2.normalized_distribution(df_train, 'SalePrice')
train["SalePrice"] = np.log1p(train["SalePrice"])
#plt.show()

########################## concatenation of train & test ################################
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Raw : 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
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

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Raw : 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''




#......................... single level (Data Handling) .................................
if single_level_Data_Handling :
    p_description += 'single level data handling : activeted\n'

    #group = ['MasVnrArea', 'MasVnrType']
    #relation = all_data[all_data['MasVnrArea'] == 0]
    #checker_v2.partial(group, relation)

    all_data.loc[688, 'MasVnrArea'] = np.floor(all_data['MasVnrArea'].mean())
    all_data.loc[1241, 'MasVnrArea'] = np.floor(all_data['MasVnrArea'].mean())
    all_data.loc[2319, 'MasVnrArea'] = np.floor(all_data['MasVnrArea'].mean())
    #print(all_data.loc[2319, 'MasVnrArea'])
    #-------------------------------------------------------------------------------------------------

    #group = ['MiscVal', 'MiscFeature']
    #relation = all_data[all_data['MiscVal'] == 0]
    #checker_v2.partial(group, relation)

    all_data.loc[873, 'MiscVal'] = 3500
    all_data.loc[1200, 'MiscVal'] = 1200
    all_data.loc[2431, 'MiscVal'] = 1200
    #print(all_data.loc[873, 'MiscVal'])
    #-------------------------------------------------------------------------------------------------

    #group = ['PoolArea', 'PoolQC']
    #relation = all_data[all_data['PoolArea'] != 0 & all_data['PoolQC'].isnull()] #'a'.isnull() = True
    #checker_v2.partial(group, relation)

    all_data.loc[2420, 'PoolQC'] = 'Fa'
    all_data.loc[2503, 'PoolQC'] = 'Gd'
    all_data.loc[2599, 'PoolQC'] = 'Fa'

    #-------------------------------------------------------------------------------------------------
    #group = ['GarageType','GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
    #         'GarageCond']
    #relation = all_data[(all_data['GarageType'] == 'Detchd') & all_data['GarageYrBlt'].isnull()]
    #checker_v2.partial(group, relation)

    all_data.loc[2126, 'GarageYrBlt'] = 1990
    all_data.loc[2576, 'GarageYrBlt'] = 1991

    all_data.loc[2126, 'GarageFinish'] = all_data['GarageFinish'].mode()[0]
    all_data.loc[2576, 'GarageFinish'] = all_data['GarageFinish'].mode()[0]
    #print(all_data.loc[2126, 'GarageFinish'])

    all_data.loc[2576, 'GarageCars'] = np.floor(all_data['GarageCars'].median())
    #print(all_data.loc[2576, 'GarageCars'])

    all_data.loc[2576, 'GarageArea'] = np.floor(all_data['GarageArea'].median())
    #print(all_data.loc[2576, 'GarageArea'])

    all_data.loc[2126, 'GarageQual'] = all_data['GarageQual'].mode()[0]
    all_data.loc[2576, 'GarageQual'] = all_data['GarageQual'].mode()[0]
    #print(all_data.loc[2576, 'GarageQual'])

    all_data.loc[2126, 'GarageCond'] = all_data['GarageCond'].mode()[0]
    all_data.loc[2576, 'GarageCond'] = all_data['GarageCond'].mode()[0]
    #print(all_data.loc[2126, 'GarageCond'])

    #-------------------------------------------------------------------------------------------------

    #group = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1',
    #        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']

    #relation = all_data
    #checker_v2.partial(group, relation)

    all_data.loc[332, 'BsmtFinType2'] = 'ALQ'
    all_data.loc[1487, 'BsmtExposure'] = 'No'
    all_data.loc[2348, 'BsmtExposure'] = 'No'
    all_data.loc[2040, 'BsmtCond'] = 'TA'
    all_data.loc[2185, 'BsmtCond'] = 'TA'
    all_data.loc[2524, 'BsmtCond'] = 'TA'
    all_data.loc[2217, 'BsmtQual'] = 'Po'
    all_data.loc[2218, 'BsmtQual'] = 'Fa'

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fixing wrong data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #--------------------------------------- Year time fixing ----------------------------------------
    #group = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
    #relation = all_data[all_data[group] >= 2019]
    #checker_v2.partial(group, relation)
    all_data.loc[2592, 'GarageYrBlt'] = 2007

    #--------------------------------------- Month time fixing ---------------------------------------
    #group = ['MoSold']
    #relation = all_data[all_data['MoSold'] > 12]
    #checker_v2.partial(group, relation)





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multi-level (Data Handling) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Raw : 4 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
if multi_level_modified == 0:
    # (categorical) converting numerical variables that are actually categorical
    # 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd'
    cols = ['MSSubClass', 'YrSold', 'MoSold']
    for var in cols:
        all_data[var] = all_data[var].astype(str)
    p_description += "cols = ['MSSubClass', 'YrSold', 'MoSold'] =  str\n"

    # 'NA' means Special value
    all_data['Functional'] = all_data['Functional'].fillna('Typ')
    all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA")

    #'NA' means most frequest value
    #'ExterQual', # categorical(numerical)
    #'ExterCond','HeatingQC','LandSlope', 'LotShape','LandContour', 'LotConfig', 'BldgType',
    #'RoofStyle', 'Foundation','SaleCondition'
    common_vars = [ 'Exterior1st', 'Exterior2nd', 'SaleType']#, 'MSSubClass', 'YearBuilt']
    for var in common_vars:
        all_data[var] = all_data[var].fillna(all_data[var].mode()[0])

    #same as construction date if no remodeling or additions
    #all_data['YearRemodAdd'] = all_data['YearRemodAdd'].fillna(all_data['YearBuilt'])
    # categorical 'NA' means 'None'
    #'MiscFeature', 'MasVnrType', 'Fence'
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
    all_data['LotFrontage'] = np.floor(all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median())))

    #same as construction date if no remodeling or additions
    all_data['YearRemodAdd'] = all_data['YearRemodAdd'].fillna(all_data['YearBuilt'])
    p_description += "all_data['YearRemodAdd'] = all_data['YearRemodAdd'].fillna(all_data['YearBuilt'])\n"

    # Collecting all object type feature and handling multi level null values
    objects = []
    for i in all_data.columns:
        if all_data[i].dtype == object:
            objects.append(i)

    all_data.update(all_data[objects].fillna('None'))

    # Collectting all numeric type feature and handling multi level null values
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in all_data.columns:
        if all_data[i].dtype in numeric_dtypes:
            numerics.append(i)
    all_data.update(all_data[numerics].fillna(0))

    if 0:
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numerics2 = []
        for i in all_data.columns:
            if all_data[i].dtype in numeric_dtypes:
                numerics2.append(i)

        # checking skew
        skew_all_data = all_data[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
        high_skew = skew_all_data[skew_all_data > 0.5]
        skew_index = high_skew.index

        for i in skew_index:
            all_data[i] = boxcox1p(all_data[i], boxcox_normmax(all_data[i] + 1))
    else:
        p_description += 'skew removed\n'

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Raw : 4 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# ///////////////////////////////////// or //////////////////////////////////////////////

if multi_level_modified:
    p_description += 'multi_level_modified : activeted\n'
    # 'NA' means Special value
    all_data['Functional'] = all_data['Functional'].fillna('Typ')
    all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA")
    #------------------------------------------------------------------------------------

    #'NA' means most frequest value
    #'ExterQual', # categorical(numerical)
    #'ExterCond','HeatingQC','LandSlope', 'LotShape','LandContour', 'LotConfig', 'BldgType',
    #'RoofStyle', 'Foundation','SaleCondition'
    common_vars = [ 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSSubClass', 'YearBuilt']
    for var in common_vars:
        all_data[var] = all_data[var].fillna(all_data[var].mode()[0])
    p_description += "'MSSubClass', 'YearBuilt' = fillna(all_data[var].mode()[0]\n"
    #------------------------------------------------------------------------------------

    # categorical 'NA' means 'None'
    #'MiscFeature', 'MasVnrType', 'Fence'
    common_vars = ['PoolQC',
                   'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in common_vars:
        all_data[col] = all_data[col].fillna('None')
    #------------------------------------------------------------------------------------

    #numerical 'NA' means 0
    common_vars = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'YrSold', 'MoSold']
    for col in common_vars:
        all_data[col] = all_data[col].fillna(0)
    p_description += "'YrSold', 'MoSold' = fillna(0)\n"
    #------------------------------------------------------------------------------------

    # (categorical) converting numerical variables that are actually categorical
    # 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd'
    cols = ['MSSubClass', 'YrSold', 'MoSold']
    for var in cols:
        all_data[var] = all_data[var].astype(str)
    p_description += "cols = ['MSSubClass', 'YrSold', 'MoSold'] =  str\n"
    #-------------------------------------------------------------------------------------

    # 'NA'means most or recent common according to base on other special groups
    all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    all_data['LotFrontage'] = np.floor(all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median())))

    #same as construction date if no remodeling or additions
    all_data['YearRemodAdd'] = all_data['YearRemodAdd'].fillna(all_data['YearBuilt'])
    p_description += "all_data['YearRemodAdd'] = all_data['YearRemodAdd'].fillna(all_data['YearBuilt'])\n"
    #------------------------------------------------------------------------------------

    # Collecting all object type feature and handling multi level null values
    objects = []
    for i in all_data.columns:
        if all_data[i].dtype == object:
            objects.append(i)

    all_data.update(all_data[objects].fillna('None'))
    #------------------------------------------------------------------------------------

    # Collectting all numeric type feature and handling multi level null values
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in all_data.columns:
        if all_data[i].dtype in numeric_dtypes:
            numerics.append(i)
    all_data.update(all_data[numerics].fillna(0))

    p_description += 'skew removed\n'



########################## Column dron & each column's value count ###############################
if column_value_use_counter:
    print('------------- count of use of each variable --------------')
    objects = []
    for i in all_data.columns:
        if all_data[i].dtype == object:
            objects.append(i)

    print('Data of Objects\n_________________')
    for i in objects:
        print(all_data[i].value_counts())
        print('\n~~~~~~~~~~~~~~~~~~~~~')

# dropping the columns which have a large amount of distance between it's value used amount
drop_columns = ['Utilities', 'Street', 'PoolQC']
print('Dropping columns : ', drop_columns)
all_data = all_data.drop([i for i in drop_columns], axis=1)
p_description += "drop_columns = ['Utilities', 'Street', 'PoolQC']\n"


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if missing_data:
    checker_v2.missing_data(all_data, 0) # checking missing data 1/0 for save as file or not
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



if o2n_converter:
    p_description += 'object to numeric converter : activeted\n'
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  fillna(0)  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # creating a set of all categorical(Ordinal) variables with a specific value to the characters
    dic = {'Grvl': 3, 'Pave': 6, 'NA': 0, 'None' : 0}
    all_data['Alley'] = checker_v2.data_converter(dic, all_data, 'Alley')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    all_data['FireplaceQu'] = checker_v2.data_converter(dic, all_data, 'FireplaceQu')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    all_data['GarageQual'] = checker_v2.data_converter(dic, all_data, 'GarageQual')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    all_data['BsmtQual'] = checker_v2.data_converter(dic, all_data, 'BsmtQual')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    all_data['GarageCond'] = checker_v2.data_converter(dic, all_data, 'GarageCond')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    all_data['BsmtCond'] = checker_v2.data_converter(dic, all_data, 'BsmtCond')

    dic = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0, 'None' : 0}
    all_data['GarageFinish'] = checker_v2.data_converter(dic, all_data, 'GarageFinish')

    dic = {'Gd': 5, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0, 'None' : 0}
    all_data['BsmtExposure'] = checker_v2.data_converter(dic, all_data, 'BsmtExposure')

    dic = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0, 'None' : 0}
    all_data['BsmtFinType1'] = checker_v2.data_converter(dic, all_data, 'BsmtFinType1')

    dic = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0, 'None' : 0}
    all_data['BsmtFinType2'] = checker_v2.data_converter(dic, all_data, 'BsmtFinType2')



    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   mod()[0]   !!!!!!!!!!!!!!!!!!!!!!!!!!
    dic = {'A': 2, 'C (all)': 3, 'FV': 1, 'I': 4, 'RH': 9, 'RL': 5, 'RP': 6, 'RM': 8}
    all_data['MSZoning'] = checker_v2.data_converter(dic, all_data, 'MSZoning')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    all_data['KitchenQual'] = checker_v2.data_converter(dic, all_data, 'KitchenQual')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    all_data['ExterQual'] = checker_v2.data_converter(dic, all_data, 'ExterQual')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    all_data['ExterCond'] = checker_v2.data_converter(dic, all_data, 'ExterCond')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    all_data['HeatingQC'] = checker_v2.data_converter(dic, all_data, 'HeatingQC')

    dic = {'Gtl': 3, 'Mod': 2, 'Sev': 1}
    all_data['LandSlope'] = checker_v2.data_converter(dic, all_data, 'LandSlope')

    dic = {'Reg': 9, 'IR1': 7, 'IR2': 5, 'IR3': 2}
    all_data['LotShape'] = checker_v2.data_converter(dic, all_data, 'LotShape')

    dic = {'Lvl': 7, 'Bnk': 6, 'HLS': 5, 'Low': 2}
    all_data['LandContour'] = checker_v2.data_converter(dic, all_data, 'LandContour')

    dic = {'Inside': 2, 'Corner': 4, 'CulDSac': 5, 'FR2': 7, 'FR3': 9}
    all_data['LotConfig'] = checker_v2.data_converter(dic, all_data, 'LotConfig')

    dic = {'1Fam': 2, '2fmCon': 4, 'Duplex': 5, 'Twnhs': 7, 'TwnhsE': 7, 'TwnhsI': 9}
    all_data['BldgType'] = checker_v2.data_converter(dic, all_data, 'BldgType')

    dic = {'Flat': 6, 'Gable': 5, 'Gambrel': 4, 'Hip': 3, 'Mansard': 2, 'Shed': 1}
    all_data['RoofStyle'] = checker_v2.data_converter(dic, all_data, 'RoofStyle')

    dic = {'BrkTil': 4, 'CBlock': 5, 'PConc': 6, 'Slab': 2, 'Stone': 3, 'Wood': 1}
    all_data['Foundation'] = checker_v2.data_converter(dic, all_data, 'Foundation')

    dic = {'Normal': 6, 'Abnorml': 5, 'AdjLand': 4, 'Alloca': 3, 'Family': 2, 'Partial': 1}
    all_data['SaleCondition'] = checker_v2.data_converter(dic, all_data, 'SaleCondition')



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Raw : 5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
############################## adding new feature ######################################

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

p_description += "['YearBuilt', 'YearRemodAdd', 'YrBltAndRemod', 'GarageYrBlt'] = nmbr\n"
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Raw :  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''





######################################## 3. Normalization handling ###############################
if normalization :
    p_description += 'Normalization : activeted\n'
    print('-------------------- Normalization ---------------------\n')
    numerics = []
    if 0:
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for i in all_data.columns:
            if all_data[i].dtype in numeric_dtypes:
                numerics.append(i)
    else:
        numerics = ['TotalSF', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'MasVnrArea']

    if save_column_name:
        # get only column names and transposes(T) row into columns
        numeric_data = all_data[numerics]
        cName_n_data = numeric_data.head(0).T
        cName_n_data.to_csv(path + 'nnormalization_save_column_names.csv')
        print('Numeric columns names saved at :' + path + 'nnormalization_save_column_names.csv')


    for i in numerics:
        if 1:
            all_data[i] = np.log1p(all_data[i])

        if 0:
            checker_v2.general_distribution(all_data, i)
            plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if save_column_name:
    # get only column names and transposes(T) row into columns
    cName_all_data = all_data.head(0).T
    cName_all_data.to_csv(path+'save_column_names.csv')
    print('Columns names saved at :'+path+'save_column_names.csv')


df_train = all_data.iloc[:len(y_train), :]
df_test = all_data.iloc[len(df_train):, :]
df_train['SalePrice'] = y_train






########################################  Heat Map : 2 ##############################################
if hit_map:
    # Complete numerical correlation matrix
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(20, 13))
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.show()

    # Partial numerical correlation matrix (salePrice)
    corr_num = 15  # number of variables for heatmap
    cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index
    corr_mat_sales = np.corrcoef(df_train[cols_corr].values.T)
    f, ax = plt.subplots(figsize=(20, 15))
    hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 7}, yticklabels=cols_corr.values,
                     xticklabels=cols_corr.values)
    plt.show()
#________________________________________________________________________________________________






#################################### creating dummy & de-couple all_data #########################
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


outliers = [30, 88, 462, 631, 1322]
final_train = final_train.drop(final_train.index[outliers])
y_train = y_train.drop(y_train.index[outliers])
# Removes columns where the threshold of zero's is (> 99.95), means has only zero values

overfit = []
for i in final_train.columns:
    counts = final_train[i].value_counts()
    zeros = counts.iloc[0]
    if ((zeros / len(final_train)) * 100) > 99.94:
        overfit.append(i)

overfit = list(overfit)
if o2n_converter == 0:
    overfit.append('MSZoning_C (all)')
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

def project_description(description):
    if file_description:
        description += '~~~~~~~~~~~~~~~~~~~~~~~ Project file data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        description += p_description
        file = open(path+'process_description.txt', file_open_order)
        file.write(description)
        print('\n____________________________________________________________________________________')
        print('process_description has been saved at : '+path+'process_description.txt')
        file.close()