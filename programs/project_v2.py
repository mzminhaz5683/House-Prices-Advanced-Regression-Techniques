# coding=utf-8
import numpy as np # -------------------linear algebra
import pandas as pd # ------------------data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # ------data manipulation
import seaborn as sns # ----------------data presentation


import warnings

from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

#Limiting floats output to 1 decimal point(s) after dot(.)
pd.set_option('display.float_format', lambda x: '{:.1f}'.format(x))
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Raw : 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
p_description = ''

# import local file
from programs import checker_v2
checker_v2.path = path = "../output/Data_observation/"
######################################## Controller ##############################################
hit_map = 0
save_column_name = 0

check_outliars_numeric = 0
check_outliars_objects = 0


save_all_data = 0
missing_data = 0

single_level_Data_Handling = 1
column_value_use_counter = 0

o2n_converter = 1

file_description = 1
file_open_order = 'w'
local_project_description = 1
######################################## design #(40) + string + #(*) = 100 ######################




######################################## START ###################################################'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Raw : 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1:
    p_description += 'SalePrice_log1p : Top\n'
    train['SalePrice'] = np.log1p(train['SalePrice'])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



############################## 2. Out-liars Handling ##################################
#..............................2a numerical analyzing...................................

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

drop_index = train[(train['TotalBsmtSF'] > 3000)].index
train = train.drop(drop_index)

p_description += "drop_index = train[(train['TotalBsmtSF'] > 3000)].index\n"
#________________________________________________________________________________________________

drop_index = train[(train['1stFlrSF'] > 2500)].index
train = train.drop(drop_index)

p_description += "drop_index = train[(train['1stFlrSF'] > 2500)].index\n"
# ________________________________________________________________________________________________

drop_index = train[(train['GrLivArea'] > 4000)].index
train = train.drop(drop_index)

p_description += "drop_index = train[(train['GrLivArea'] > 4000)].index\n"
# ________________________________________________________________________________________________
p_description += '-----------------------------------------------------------------\n'
#________________________________________________________________________________________________
#________________________________________________________________________________________________




#...............................2b categorical analyzing.................................
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



########################## concatenation of train & test ################################
y_train = train.SalePrice.reset_index(drop=True) # assign
df_train = train.drop(['SalePrice'], axis = 1) # drop
df_test = test # assign


all_data = pd.concat([df_train, df_test]).reset_index(drop=True) # concatenation
#dtypes = all_data.dtypes


if save_all_data:
    all_data.to_csv(path+'all_data_prime.csv')
    print('\nAll prime data has been saved at : '+path+'all_data_prime.csv')


print('____________________________________________________________________________________')
print('all_data shape (Rows, Columns) & Columns-(ID, SalePrice): ', all_data.shape)




############################## missing data operation ##################################
if missing_data:
    checker_v2.missing_data(all_data, 0) # checking missing data 1/0 for save as file or not



#......................... single level (Data Handling) .................................
if single_level_Data_Handling :
    p_description += 'single level data handling : activeted\n'

    #group = ['MasVnrArea', 'MasVnrType']
    #relation = all_data[all_data['MasVnrArea'] == 0]
    #checker_v2.partial(group, relation)

    all_data.loc[687, 'MasVnrArea'] = np.floor(all_data['MasVnrArea'].mean())
    all_data.loc[1240, 'MasVnrArea'] = np.floor(all_data['MasVnrArea'].mean())
    all_data.loc[2317, 'MasVnrArea'] = np.floor(all_data['MasVnrArea'].mean())

    #relation = all_data[all_data['MasVnrArea'] == 0]
    #checker_v2.partial(group, relation)
    #-------------------------------------------------------------------------------------------------

    #group = ['MiscVal', 'MiscFeature']
    #relation = all_data[all_data['MiscVal'] == 0]
    #checker_v2.partial(group, relation)

    all_data.loc[872, 'MiscVal'] = 3500
    all_data.loc[1199, 'MiscVal'] = 1200
    all_data.loc[2429, 'MiscVal'] = 1200

    #relation = all_data[all_data['MiscVal'] == 0]
    #checker_v2.partial(group, relation)
    #-------------------------------------------------------------------------------------------------

    #group = ['PoolArea', 'PoolQC']
    #relation = all_data[all_data['PoolArea'] != 0 & all_data['PoolQC'].isnull()] #'a'.isnull() = True
    #checker_v2.partial(group, relation)

    all_data.loc[2418, 'PoolQC'] = 'Fa'
    all_data.loc[2501, 'PoolQC'] = 'Gd'
    all_data.loc[2597, 'PoolQC'] = 'Fa'

    #relation = all_data[all_data['PoolArea'] != 0 & all_data['PoolQC'].isnull()]  # 'a'.isnull() = True
    #checker_v2.partial(group, relation)
    #-------------------------------------------------------------------------------------------------

    #group = ['GarageType','GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']
    #relation = all_data[(all_data['GarageType'] == 'Detchd') & all_data['GarageYrBlt'].isnull()]
    #checker_v2.partial(group, relation)

    all_data.loc[2124, 'GarageYrBlt'] = 1990
    all_data.loc[2574, 'GarageYrBlt'] = 1991

    all_data.loc[2124, 'GarageFinish'] = all_data['GarageFinish'].mode()[0]
    all_data.loc[2574, 'GarageFinish'] = all_data['GarageFinish'].mode()[0]


    all_data.loc[2574, 'GarageCars'] = np.floor(all_data['GarageCars'].median())


    all_data.loc[2574, 'GarageArea'] = np.floor(all_data['GarageArea'].median())


    all_data.loc[2124, 'GarageQual'] = all_data['GarageQual'].mode()[0]
    all_data.loc[2574, 'GarageQual'] = all_data['GarageQual'].mode()[0]


    all_data.loc[2124, 'GarageCond'] = all_data['GarageCond'].mode()[0]
    all_data.loc[2574, 'GarageCond'] = all_data['GarageCond'].mode()[0]

    #relation = all_data[(all_data['GarageType'] == 'Detchd') & all_data['GarageYrBlt'].isnull()]
    #checker_v2.partial(group, relation)
    #-------------------------------------------------------------------------------------------------

    #group = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    #relation = all_data
    #checker_v2.partial(group, relation)

    all_data.loc[332, 'BsmtFinType2'] = 'ALQ'

    all_data.loc[947, 'BsmtExposure'] = 'No'
    all_data.loc[1485, 'BsmtExposure'] = 'No'
    all_data.loc[2346, 'BsmtExposure'] = 'No'

    all_data.loc[2038, 'BsmtCond'] = 'TA'
    all_data.loc[2183, 'BsmtCond'] = 'TA'
    all_data.loc[2522, 'BsmtCond'] = 'TA'

    all_data.loc[2215, 'BsmtQual'] = 'Po'
    all_data.loc[2216, 'BsmtQual'] = 'Fa'

    #relation = all_data
    #checker_v2.partial(group, relation)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fixing wrong data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #--------------------------------------- Year time fixing ----------------------------------------
    #group = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
    #relation = all_data[all_data[group] >= 2019] # 'null' > 2019
    #checker_v2.partial(group, relation)

    all_data.loc[2590, 'GarageYrBlt'] = 2007

    #relation = all_data[all_data[group] >= 2019] # 'null' > 2019
    #checker_v2.partial(group, relation)


    #--------------------------------------- Month time fixing ---------------------------------------
    #group = ['MoSold']
    #relation = all_data[all_data['MoSold'] > 12]
    #checker_v2.partial(group, relation)
else:
    p_description += 'single level data handling : deactiveted\n'


#~~~~~~~~~~~~~~~~~~~~~~~~ Multi-level (Data Handling) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# 'NA' means Special value
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")
all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA")
#------------------------------------------------------------------------------------


# categorical( = need to be numerical)
c2n = ['Street','LotShape','LandContour','Utilities','LotConfig',
       'LandSlope','BldgType','RoofStyle','ExterQual','ExterCond',
       'Foundation','HeatingQC','SaleCondition']
#'NA' means most frequest value
common_vars = [ 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSSubClass', 'YearBuilt']
common_vars += c2n
for var in common_vars:
    all_data[var] = all_data[var].fillna(all_data[var].mode()[0])
p_description += "common_vars += c2n &\n" \
                 "'MSSubClass', 'YearBuilt' = fillna(all_data[var].mode()[0])\n"
#------------------------------------------------------------------------------------


# categorical( = need to be numerical)
common_vars = ['Alley','BsmtQual','BsmtCond','BsmtExposure',
               'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType',
               'GarageFinish','GarageQual','GarageCond','PoolQC','GarageType']
# categorical 'NA' means 'None'
for col in common_vars:
    all_data[col] = all_data[col].fillna('None')
#------------------------------------------------------------------------------------


# numerical 'NA' means 0
common_vars = ['YrSold', 'MoSold', 'GarageYrBlt', 'GarageArea', 'GarageCars']
for col in common_vars:
    all_data[col] = all_data[col].fillna(0)
p_description += "'YrSold', 'MoSold' = fillna(0)\n"
#------------------------------------------------------------------------------------


# 'NA'means most or recent common value according to (base on) other special groups
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
all_data['LotFrontage'] = np.floor(all_data.groupby('BldgType')['LotFrontage'].transform(lambda x: x.fillna(x.median())))

# condition of data description
all_data['YearRemodAdd'] = all_data['YearRemodAdd'].fillna(all_data['YearBuilt'])
p_description += "all_data['YearRemodAdd'] = all_data['YearRemodAdd'].fillna(all_data['YearBuilt'])\n"
#-------------------------------------------------------------------------------------


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



####################### conversion of data-type ####################################

# (categorical) converting numerical variables that are actually categorical
# 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd'
cols = ['MSSubClass', 'OverallQual', 'OverallCond', 'YrSold', 'MoSold']#, 'YearBuilt', 'YearRemodAdd']
for var in cols:
    all_data[var] = all_data[var].astype(str)
p_description += "cols = {0} =  str\n".format(cols)
#---------------------------------------------------------------------------------


p_description += 'skew removed\n'
if 0:
    all_data.to_csv(path+'all_data_secondary.csv')
    print('\nAll modified data has been saved at : '+path+'all_data_secondary.csv')


################### column_value_use_count & column drop ###############################
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
#drop_columns = ['Street', 'Utilities', 'PoolQC']
drop_columns = ['Street','Utilities','Condition2','RoofMatl','Heating','PoolQC']
print('Dropping columns : ', drop_columns)
all_data = all_data.drop([i for i in drop_columns], axis=1)
p_description += "drop_columns = {0}\n".format(drop_columns)



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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if missing_data:
    checker_v2.missing_data(all_data, 0) # checking missing data 1/0 for save as file or not
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



####################### skewed & log1p on numerical features ###########################################

'''Extract numeric variables merged data.'''
df_merged_num = all_data.select_dtypes(include = ['int64', 'float64'])


'''Make the tranformation of the explanetory variables'''
df_merged_skewed = np.log1p(df_merged_num[df_merged_num.skew()[df_merged_num.skew() > 0.5].index])

# Normal variables
df_merged_normal = df_merged_num[df_merged_num.skew()[df_merged_num.skew() < 0.5].index]

# Merging
df_merged_num_all = pd.concat([df_merged_skewed, df_merged_normal], axis=1)

'''Update numerical variables with transformed variables.'''
df_merged_num.update(df_merged_num_all)

'''Creating scaler object.'''
scaler = RobustScaler()

'''Fit scaler object on train data.'''
scaler.fit(df_merged_num)

'''Apply scaler object to both train and test data.'''
df_merged_num_scaled = scaler.transform(df_merged_num)

'''Retrive column names'''
df_merged_num_scaled = pd.DataFrame(data = df_merged_num_scaled, columns = df_merged_num.columns, index = df_merged_num.index)
# Pass the index of index df_merged_num, otherwise it will sum up the index.



######################### categorical ################################################

"""Let's extract categorical variables first and convert them into category."""
df_merged_cat = all_data.select_dtypes(include = ['object']).astype('category')



if o2n_converter:
    p_description += 'object to numeric converter : activeted\n'
    # !!!!!!!!!!!!!!!!!!!!!!!!!  fillna(0)  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # creating a set of all categorical(Ordinal) variables with a specific value to the characters
    dic = {'Grvl': 3, 'Pave': 6, 'NA': 0, 'None' : 0}
    df_merged_cat['Alley'] = checker_v2.data_converter(dic, df_merged_cat, 'Alley')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['FireplaceQu'] = checker_v2.data_converter(dic, df_merged_cat, 'FireplaceQu')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['GarageQual'] = checker_v2.data_converter(dic, df_merged_cat, 'GarageQual')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['BsmtQual'] = checker_v2.data_converter(dic, df_merged_cat, 'BsmtQual')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['GarageCond'] = checker_v2.data_converter(dic, df_merged_cat, 'GarageCond')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['BsmtCond'] = checker_v2.data_converter(dic, df_merged_cat, 'BsmtCond')

    dic = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['GarageFinish'] = checker_v2.data_converter(dic, df_merged_cat, 'GarageFinish')

    dic = {'Gd': 5, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['BsmtExposure'] = checker_v2.data_converter(dic, df_merged_cat, 'BsmtExposure')

    dic = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['BsmtFinType1'] = checker_v2.data_converter(dic, df_merged_cat, 'BsmtFinType1')

    dic = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0, 'None' : 0}
    df_merged_cat['BsmtFinType2'] = checker_v2.data_converter(dic, df_merged_cat, 'BsmtFinType2')



    # !!!!!!!!!!!!!!!!!!!!!!!!!!!   mod()[0]   !!!!!!!!!!!!!!!!!!!!!!!!!!
    dic = {'A': 2, 'C (all)': 3, 'FV': 1, 'I': 4, 'RH': 9, 'RL': 5, 'RP': 6, 'RM': 8}
    df_merged_cat['MSZoning'] = checker_v2.data_converter(dic, df_merged_cat, 'MSZoning')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df_merged_cat['KitchenQual'] = checker_v2.data_converter(dic, df_merged_cat, 'KitchenQual')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df_merged_cat['ExterQual'] = checker_v2.data_converter(dic, df_merged_cat, 'ExterQual')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df_merged_cat['ExterCond'] = checker_v2.data_converter(dic, df_merged_cat, 'ExterCond')

    dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df_merged_cat['HeatingQC'] = checker_v2.data_converter(dic, df_merged_cat, 'HeatingQC')

    dic = {'Gtl': 3, 'Mod': 2, 'Sev': 1}
    df_merged_cat['LandSlope'] = checker_v2.data_converter(dic, df_merged_cat, 'LandSlope')

    dic = {'Reg': 9, 'IR1': 7, 'IR2': 5, 'IR3': 2}
    df_merged_cat['LotShape'] = checker_v2.data_converter(dic, df_merged_cat, 'LotShape')

    dic = {'Lvl': 7, 'Bnk': 6, 'HLS': 5, 'Low': 2}
    df_merged_cat['LandContour'] = checker_v2.data_converter(dic, df_merged_cat, 'LandContour')

    dic = {'Inside': 2, 'Corner': 4, 'CulDSac': 5, 'FR2': 7, 'FR3': 9}
    df_merged_cat['LotConfig'] = checker_v2.data_converter(dic, df_merged_cat, 'LotConfig')

    dic = {'1Fam': 2, '2fmCon': 4, 'Duplex': 5, 'Twnhs': 7, 'TwnhsE': 7, 'TwnhsI': 9}
    df_merged_cat['BldgType'] = checker_v2.data_converter(dic, df_merged_cat, 'BldgType')

    dic = {'Flat': 6, 'Gable': 5, 'Gambrel': 4, 'Hip': 3, 'Mansard': 2, 'Shed': 1}
    df_merged_cat['RoofStyle'] = checker_v2.data_converter(dic, df_merged_cat, 'RoofStyle')

    dic = {'BrkTil': 4, 'CBlock': 5, 'PConc': 6, 'Slab': 2, 'Stone': 3, 'Wood': 1}
    df_merged_cat['Foundation'] = checker_v2.data_converter(dic, df_merged_cat, 'Foundation')

    dic = {'Normal': 6, 'Abnorml': 5, 'AdjLand': 4, 'Alloca': 3, 'Family': 2, 'Partial': 1}
    df_merged_cat['SaleCondition'] = checker_v2.data_converter(dic, df_merged_cat, 'SaleCondition')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    '''All the encodeded variables have int64 dtype except OverallQual and OverallCond. So convert them back into int64.'''
    df_merged_cat.loc[:, ['OverallQual', 'OverallCond']] = df_merged_cat.loc[:, ['OverallQual', 'OverallCond']].astype('int64')

    '''Extract label encoded variables'''
    df_merged_label_encoded = df_merged_cat.select_dtypes(include=['int64'])

    '''Now selecting the nominal vaiables for one hot encording'''
    df_merged_one_hot = df_merged_cat.select_dtypes(include=['category'])

    """Let's get the dummies variable"""
    df_merged_one_hot = pd.get_dummies(df_merged_one_hot).reset_index(drop=True)

    """Let's concat one hot encoded and label encoded variable together"""
    df_merged_encoded = pd.concat([df_merged_one_hot, df_merged_label_encoded], axis=1)


else:
    p_description += 'object to numeric converter : deactiveted\n'
    df_merged_encoded = pd.get_dummies(df_merged_cat).reset_index(drop=True)



############################# join numeric & categoric #############################
'''Finally join processed categorical and numerical variables'''
all_data = pd.concat([df_merged_num_scaled, df_merged_encoded], axis=1)

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
final_train = df_train.drop(['SalePrice'], axis = 1)
final_test = df_test

overfit = []
for i in final_train.columns:
    counts = final_train[i].value_counts()
    zeros = counts.iloc[0]
    if ((zeros / len(final_train)) * 100) > 99.94:
        overfit.append(i)

overfit = list(overfit)

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
if local_project_description:
    project_description('')