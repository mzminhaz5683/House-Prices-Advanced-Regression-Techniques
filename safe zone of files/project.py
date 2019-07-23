import pandas as pd # data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # data manipulation
import seaborn as sns # data presentation
import numpy as np # linear algebra
from scipy.stats import norm #for some statistics
from scipy import stats  # scientific notation handler
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
#%matplotlib inline
from programs import checker # import local file

#.....Importing & Checking Inputs.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print("df_train input size(+ID):", df_train.shape) #1460 samples, 80 features +1(ID)
print("df_test input size(+ID):", df_test.shape) # 1459 test cases, 79 features +1(ID)

#.....Dropping 'Id' column since it's not a necessary item on prediction
df_train_ID = df_train['Id']
df_test_ID = df_test['Id']
df_train.drop('Id', axis=1, inplace=True)
df_test.drop('Id', axis=1, inplace=True)

###########################################  Heat Map  ##################################################
'''
# Numerical values correlation matrix, to locate dependencies between different variables.
# Complete numerical correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(20, 13))
sns.heatmap(corrmat, vmax=1, square=True)
plt.show()

# Partial numerical correlation matrix (salePrice)
corr_num = 15 #number of variables for heatmap
cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index
corr_mat_sales = np.corrcoef(df_train[cols_corr].values.T)
f, ax = plt.subplots(figsize=(15, 11))
hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
#plt.show()
'''
###################################### 1. Data Handling ##################################################

#.........................................missing data observing.........................................
ntrain = df_train.shape[0]
y_train = df_train.SalePrice.values

df_train.drop(['SalePrice'], axis = 1, inplace = True)
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

total = all_data.isnull().sum().sort_values(ascending=False)
percent = ((all_data.isnull().sum()/all_data.isnull().count()) * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data)


#.........................................dealing with missing data.....................................

# categorical 'NA' means 'None'
common_vars = ['MiscFeature', 'MasVnrType', 'GarageType', 'Fence']
for col in common_vars:
    all_data[col] = all_data[col].fillna('None')


# categorical(numerical) 'NA' means 0
common_vars = ['PoolQC', 'Alley', 'FireplaceQu', 'GarageQual','BsmtQual','GarageCond', # categorical(numerical)
               'BsmtCond', 'GarageFinish', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF2','BsmtFinType2',
               'OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF', # numerical
               'FullBath','TotRmsAbvGrd','Fireplaces','MasVnrArea','BsmtFinSF1','LotFrontage','2ndFlrSF',
               'BsmtFullBath','WoodDeckSF','OpenPorchSF','BsmtHalfBath','MSSubClass','BsmtUnfSF',
               'YrSold', 'MoSold', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd']
#'MSSubClass','YrSold','MoSold','GarageYrBlt','YearBuilt','YearRemodAdd' >> special categorical
for col in common_vars:
    all_data[col] = all_data[col].fillna(0)


#'NA' means the most common string
common_vars = ['Exterior2nd', 'Exterior1st', 'SaleType', 'Electrical', # categorical
               'MSZoning', 'Utilities', 'KitchenQual', 'ExterQual','Street', # categorical(numerical)
               'ExterCond','HeatingQC','LandSlope', 'LotShape','LandContour', 'LotConfig', 'BldgType',
               'RoofStyle', 'Foundation','SaleCondition']
for var in common_vars:
    all_data[var] = all_data[var].fillna(all_data[var].mode()[0])

# (categorical) Functional: 'NA' means 'typical' variable
all_data['Functional'] = all_data['Functional'].fillna('Typ')


#.....................................Data conversion...................................................

# missing in 'LotFrontage' are the median of 'LotFrontage' of the group of the belonging 'Neighborhood'.
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# (categorical) converting numerical variables that are actually categorical
#'MSSubClass','YrSold','MoSold','GarageYrBlt','YearBuilt','YearRemodAdd' >> special categorical
cols = ['MSSubClass', 'YrSold', 'MoSold', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd']
for var in cols:
    all_data[var] = all_data[var].astype(str)


# (categorical) creating a set of all categorical variables
cols = [ 'GarageType', 'Functional', 'Fence', 'PavedDrive', 'CentralAir', 'MSSubClass', 'Neighborhood',
         'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
         'Heating', 'Electrical', 'MiscFeature', 'SaleType', 'YrSold', 'MoSold', 'GarageYrBlt',
         'YearBuilt', 'YearRemodAdd']

# process columns, apply LabelEncoder to categorical features
from sklearn.preprocessing import LabelEncoder #import labelEncoder to process data
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  fillna(0)  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# creating a set of all categorical(Ordinal) variables with a specific value to the characters
dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'NA':0}
all_data['PoolQC'] = checker.data_converter(dic, all_data, 'PoolQC')

dic = {'Grvl':3, 'Pave':6, 'NA':0}
all_data['Alley'] = checker.data_converter(dic, all_data, 'Alley')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1, 'NA':0}
all_data['FireplaceQu'] = checker.data_converter(dic, all_data, 'FireplaceQu')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1, 'NA':0}
all_data['GarageQual'] = checker.data_converter(dic, all_data, 'GarageQual')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1, 'NA':0}
all_data['BsmtQual'] = checker.data_converter(dic, all_data, 'BsmtQual')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1, 'NA':0}
all_data['GarageCond'] = checker.data_converter(dic, all_data, 'GarageCond')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1, 'NA':0}
all_data['BsmtCond'] = checker.data_converter(dic, all_data, 'BsmtCond')

dic = {'Fin':3,'RFn':2, 'Unf':1, 'NA':0}
all_data['GarageFinish'] = checker.data_converter(dic, all_data, 'GarageFinish')

dic = {'Gd':5,'Av':3,'Mn':2, 'No':1, 'NA':0}
all_data['BsmtExposure'] = checker.data_converter(dic, all_data, 'BsmtExposure')

dic = {'GLQ':6, 'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2, 'Unf':1, 'NA':0}
all_data['BsmtFinType1'] = checker.data_converter(dic, all_data, 'BsmtFinType1')

dic = {'GLQ':6, 'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2, 'Unf':1, 'NA':0}
all_data['BsmtFinType2'] = checker.data_converter(dic, all_data, 'BsmtFinType2')

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   mod()[0]   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

dic = {'A':2, 'C (all)':3, 'FV':1, 'I':4, 'RH':9, 'RL':5, 'RP':6,'RM':8}
all_data['MSZoning'] = checker.data_converter(dic, all_data, 'MSZoning')

dic = {'AllPub':9, 'NoSewr':8, 'NoSeWa':7,'ELO':5}
all_data['Utilities'] = checker.data_converter(dic, all_data, 'Utilities')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1}
all_data['KitchenQual'] = checker.data_converter(dic, all_data, 'KitchenQual')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1}
all_data['ExterQual'] = checker.data_converter(dic, all_data, 'ExterQual')

dic = {'Grvl':7, 'Pave':9}
all_data['Street'] = checker.data_converter(dic, all_data, 'Street')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1}
all_data['ExterCond'] = checker.data_converter(dic, all_data, 'ExterCond')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1}
all_data['HeatingQC'] = checker.data_converter(dic, all_data, 'HeatingQC')

dic = {'Gtl':3, 'Mod':2,'Sev':1}
all_data['LandSlope'] = checker.data_converter(dic, all_data, 'LandSlope')

dic = {'Reg':9,'IR1':7,'IR2':5, 'IR3':2}
all_data['LotShape'] = checker.data_converter(dic, all_data, 'LotShape')

dic = {'Lvl':7,'Bnk':6,'HLS':5, 'Low':2}
all_data['LandContour'] = checker.data_converter(dic, all_data, 'LandContour')

dic = {'Inside':2, 'Corner':4, 'CulDSac':5, 'FR2':7, 'FR3':9}
all_data['LotConfig'] = checker.data_converter(dic, all_data, 'LotConfig')

dic = {'1Fam':2, '2fmCon':4, 'Duplex':5, 'Twnhs':7, 'TwnhsE':7, 'TwnhsI':9}
all_data['BldgType'] = checker.data_converter(dic, all_data, 'BldgType')

dic = {'Flat':6, 'Gable':5, 'Gambrel':4, 'Hip':3, 'Mansard':2, 'Shed':1}
all_data['RoofStyle'] = checker.data_converter(dic, all_data, 'RoofStyle')

dic = {'BrkTil':4, 'CBlock':5, 'PConc':6, 'Slab':2, 'Stone':3, 'Wood':1}
all_data['Foundation'] = checker.data_converter(dic, all_data, 'Foundation')

dic = {'Normal':6, 'Abnorml':5, 'AdjLand':4, 'Alloca':3, 'Family':2, 'Partial':1}
all_data['SaleCondition'] = checker.data_converter(dic, all_data, 'SaleCondition')
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#  Adding total sqfootage feature
all_data['Total_SF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#  Adding total bathrooms feature
all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))
#  Adding total porch sqfootage feature
all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
                              all_data['WoodDeckSF'])

#print(all_data.shape)
###########################################  Heat Map  ##################################################
'''
df_train_t = all_data[:ntrain] # create temporary df_train
df_train_t['SalePrice'] = y_train #adding 'SalePrice' column into temporary df_train
df_train_t['SalePrice'] = y_train
# Numerical values correlation matrix, to locate dependencies between different variables.
# Partial numerical correlation matrix (salePrice)
corrmat = df_train_t.corr()
corr_num = 15 #number of variables for heatmap
cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index
corr_mat_sales = np.corrcoef(df_train_t[cols_corr].values.T)
f, ax = plt.subplots(figsize=(15, 11))
hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
plt.show()
'''
# separate all_data into df_train & df_test
df_train = all_data[:ntrain]
df_test = all_data[ntrain:]
df_train['SalePrice'] = y_train #adding 'SalePrice' column into df_train
#print(df_train.isnull().sum().max()) #just checking that there's no missing data missing

cName_train = df_train.head(0).T # get only column names and transposes(T) row into columns
cName_train.to_csv('../output/column_name_df_train_v2.csv') # column names after handling missing data
df_train.to_csv('../output/df_train_v2.csv')
#print("df_train set size after handling missing data(-ID):", df_train.shape) # 1460 samples, 79 features
#print("df_test set size after handling missing data(-ID):", df_test.shape) # 1459 samples, 78 features

##################################### 2. Out-liars Handling #############################################
#.......................................2a numerical analyzing.......................................

#checker.numerical_relationship(df_train, 'OverallQual')

#checker.numerical_relationship(df_train, 'Total_SF')
drop_index = df_train[(df_train['Total_SF'] > 7500) & (df_train['SalePrice']<200000)].index
df_train = df_train.drop(drop_index)
#checker.numerical_relationship(df_train, 'Total_SF')

#checker.numerical_relationship(df_train, 'GrLivArea')
drop_index = df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice']<300000)].index
df_train = df_train.drop(drop_index)
#checker.numerical_relationship(df_train, 'GrLivArea')
#checker.numerical_relationship(df_train, 'GarageCars')
#checker.numerical_relationship(df_train, 'Total_Bathrooms')

#checker.numerical_relationship(df_train, 'GarageArea')
drop_index = df_train[(df_train['GarageArea'] > 1200) & (df_train['SalePrice']<300000)].index
df_train = df_train.drop(drop_index)
#checker.numerical_relationship(df_train, 'GarageArea')
#checker.numerical_relationship(df_train, 'TotalBsmtSF')
#checker.numerical_relationship(df_train, '1stFlrSF')
#checker.numerical_relationship(df_train, 'FullBath')
#checker.numerical_relationship(df_train, 'TotRmsAbvGrd') ~~~~
#checker.numerical_relationship(df_train, 'Fireplaces')

#checker.numerical_relationship(df_train, 'MasVnrArea')
drop_index = df_train[(df_train['MasVnrArea'] > 1500) & (df_train['SalePrice']<300000)].index
df_train = df_train.drop(drop_index)
#checker.numerical_relationship(df_train, 'MasVnrArea')
#checker.numerical_relationship(df_train, 'BsmtFinSF1')

#checker.numerical_relationship(df_train, 'LotFrontage')
drop_index = df_train[(df_train['LotFrontage'] > 250) & (df_train['SalePrice']<300000)].index
df_train = df_train.drop(drop_index)
#checker.numerical_relationship(df_train, 'LotFrontage')
#checker.numerical_relationship(df_train, '2ndFlrSF')
#checker.numerical_relationship(df_train, 'BsmtFullBath') ~~~
#checker.numerical_relationship(df_train, 'WoodDeckSF')

#checker.numerical_relationship(df_train, 'OpenPorchSF')
drop_index = df_train[(df_train['OpenPorchSF'] > 500) & (df_train['SalePrice']<40000)].index
df_train = df_train.drop(drop_index)
#checker.numerical_relationship(df_train, 'OpenPorchSF')

#.......................................2b categorical analyzing.......................................


#checker.categorical_relationship(df_train, 'YearBuilt')
drop_index = df_train[(df_train['SalePrice']>600000)].index
df_train = df_train.drop(drop_index)
#checker.categorical_relationship(df_train, 'YearBuilt')

#checker.categorical_relationship(df_train, 'GarageYrBlt')
drop_index = df_train[(df_train['SalePrice']>600000)].index
df_train = df_train.drop(drop_index)
#checker.categorical_relationship(df_train, 'GarageYrBlt')

#checker.categorical_relationship(df_train, 'YearRemodAdd')
drop_index = df_train[(df_train['SalePrice']>600000)].index
df_train = df_train.drop(drop_index)
#checker.categorical_relationship(df_train, 'YearRemodAdd')


###################################### 3. Normalization handling #########################################

#checker.general_distribution(df_train, 'SalePrice')
#plt.show()
#checker.normalized_distribution(df_train, 'SalePrice')
#plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#checker.general_distribution(df_train, 'Total_SF')
#plt.show()
#checker.normalized_distribution(df_train, 'Total_SF')
#plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#checker.general_distribution(df_train, 'GrLivArea')
#plt.show()
#checker.normalized_distribution(df_train, 'GrLivArea')
#plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#checker.general_distribution(df_train, 'TotalBsmtSF')
#plt.show()
#checker.normalized_distribution(df_train, 'TotalBsmtSF')
#plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#checker.general_distribution(df_train, '1stFlrSF')
#plt.show()
#checker.normalized_distribution(df_train, '1stFlrSF')
#plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#checker.general_distribution(df_train, 'MasVnrArea')
#plt.show()
#checker.normalized_distribution(df_train, 'MasVnrArea')
#plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#checker.general_distribution(df_train, 'BsmtFinSF1')
#plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#checker.general_distribution(df_train, '2ndFlrSF')
#plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#checker.general_distribution(df_train, 'WoodDeckSF')
#plt.show()
#checker.normalized_distribution(df_train, 'WoodDeckSF')
#plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#checker.general_distribution(df_train, 'OpenPorchSF')
#plt.show()
#checker.normalized_distribution(df_train, 'OpenPorchSF')
#plt.show()
################################### 3. Homoscedasticity handling #########################################


############################ 4. Converting categorical variable into dummy ###############################
ntrain = df_train.shape[0] # load df_train size
y_train = df_train.SalePrice.values # allocate SalePrice into y_train
df_train.drop(['SalePrice'], axis = 1, inplace = True) # drop SalePrice

print("df_train final size:", df_train.shape) # 1460 samples, 79 features
print("df_test final size:", df_test.shape) # 1459 samples, 78 features


all_data = pd.concat((df_train, df_test)).reset_index(drop=True) #concat to make dummy
all_data = pd.get_dummies(all_data) # dummy

df_train = all_data[:ntrain]
df_test = all_data[ntrain:]
# Removes columns where the threshold of zero's is (> 99.95), means has only zero values
overfit = []
for i in df_train.columns:
    counts = df_train[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(df_train) * 100 > 99.95:
        overfit.append(i)

overfit = list(overfit)
df_train = df_train.drop(overfit, axis=1).copy()
df_test = df_test.drop(overfit, axis=1).copy()

print('final shape (df_train, y_train, df_test): ',df_train.shape,y_train.shape,df_test.shape)
# main shape (df_train, y_train, df_test):  (1448, 139) (1448,) (1459, 139)

def get_train_label():
    print("y_train of get_train_label():", y_train.shape)
    return y_train

def get_test_ID():
    print("df_test_ID of get_test_ID():", df_test_ID.shape)  # 1460 samples
    return df_test_ID

def get_train_test_data():
    print('final shape of get_train_test_data(): ', df_train.shape, y_train.shape, df_test.shape)
    # main shape (df_train, y_train, df_test):  (1448, 139) (1448,) (1459, 139)
    return df_train, df_test

'''
df_train['SalePrice'] = y_train
df_train['Id'] = df_train_ID
df_test['Id'] = df_test_ID
#X = df_train.drop(overfit, axis=1).copy()
#X_test = df_test.drop(overfit, axis=1).copy()

print('After dummy')
print("df_train set size (+ID, +SalePrice):", df_train.shape) #1460 samples
print("df_test set size (+ID):", df_test.shape) # 1459 df_test cases
#df_train.to_csv('../output/processed_train.csv')
#df_test.to_csv('../output/processed_test.csv')
'''