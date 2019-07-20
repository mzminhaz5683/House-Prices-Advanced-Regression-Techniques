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
print("df_train set size:", df_train.shape) #1460 samples
print("df_test set size:", df_test.shape) # 1459 df_test cases

#.....Dropping 'Id' column since it's not a necessary item on prediction
df_train_ID = df_train['Id']
df_test_ID = df_test['Id']
df_train.drop(['Id'], axis=1, inplace=True)
df_test.drop(['Id'], axis=1, inplace=True)


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


# GarageType, GarageFinish, GarageQual and GarageCond represent similar & NA means "None"
for col in ('GarageQual', 'GarageType', 'GarageFinish', 'GarageCond'):
    all_data[col] = all_data[col].fillna("None")

# GarageYrBlt, GarageArea and GarageCars : NA means 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath:
# "NA" means 0 for no basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2' represent similar
# categorical meaning NA means 'None'
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

# masonry veneer: 0 for area and None for category
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


# Functional: NA means typical
all_data['Functional'] = all_data['Functional'].fillna('Typ')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~set the most common string by ''all_data[cell].mode()[0])''

# Utilises: won't help in predictive modelling
#all_data.drop(['Utilities'], axis = 1, inplace = True) #?????????????????????????????????????????
# Utilities: No for No utility
all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])

# MSZoning: NA replace most common value of the list "RL"
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# Electrical: NA means SBrkr
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


#SaleType: NA means WD
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# KitchenQual
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# Exterior1st and Exterior2nd: NA means most common string
all_data['ExterQual'] = all_data['ExterQual'].fillna(all_data['ExterQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

#.....................................Data conversion...................................................


# most important
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# converting numerical variables that are actually categorical
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
#all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# creating a set of all categorical variables
cols = ('GarageType', 'GarageFinish', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Functional', 'Fence', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features
from sklearn.preprocessing import LabelEncoder #import labelEncoder to process data
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# creating a set of all categorical(Ordinal) variables with a specific value to the characters
col2 = ('PoolQC', 'Alley', 'FireplaceQu', 'GarageQual', 'GarageCond', 'MSZoning', 'BsmtQual',
        'BsmtCond', 'Utilities', 'KitchenQual', 'ExterQual', 'ExterCond', 'HeatingQC',
        'LandSlope', 'LotShape')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'None':0}
all_data['PoolQC'] = checker.data_converter(dic, all_data, 'PoolQC')

dic = {'Pave':6, 'Grvl':3, 'None':0}
all_data['Alley'] = checker.data_converter(dic, all_data, 'Alley')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1, 'None':0}
all_data['FireplaceQu'] = checker.data_converter(dic, all_data, 'FireplaceQu')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1, 'None':0}
all_data['GarageQual'] = checker.data_converter(dic, all_data, 'GarageQual')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1, 'None':0}
all_data['GarageCond'] = checker.data_converter(dic, all_data, 'GarageCond')

dic = {'RM':8, 'RP':6, 'RL':5, 'RH':9, 'I':4, 'FV':1, 'C (all)':3,'A':2}
all_data['MSZoning'] = checker.data_converter(dic, all_data, 'MSZoning')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1, 'None':0}
all_data['BsmtQual'] = checker.data_converter(dic, all_data, 'BsmtQual')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1, 'None':0}
all_data['BsmtCond'] = checker.data_converter(dic, all_data, 'BsmtCond')

dic = {'AllPub':9, 'NoSewr':8, 'NoSeWa':7,'ELO':5}
all_data['Utilities'] = checker.data_converter(dic, all_data, 'Utilities')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1}
all_data['KitchenQual'] = checker.data_converter(dic, all_data, 'KitchenQual')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1}
all_data['ExterQual'] = checker.data_converter(dic, all_data, 'ExterQual')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1}
all_data['ExterCond'] = checker.data_converter(dic, all_data, 'ExterCond')

dic = {'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po':1}
all_data['HeatingQC'] = checker.data_converter(dic, all_data, 'HeatingQC')

dic = {'Gtl':3, 'Mod':2,'Sev':1}
all_data['LandSlope'] = checker.data_converter(dic, all_data, 'LandSlope')

dic = {'Reg':9,'IR1':7,'IR2':5, 'IR3':2}
all_data['LotShape'] = checker.data_converter(dic, all_data, 'LotShape')

#  Adding total sqfootage feature
all_data['Total_SF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#  Adding total bathrooms feature
all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))
#  Adding total porch sqfootage feature
all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
                              all_data['WoodDeckSF'])

# Not normally distributed can not be normalised and duplicated has no central tendency
all_data = all_data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                          'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
                          'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF',
                          'MasVnrArea', 'BsmtFinSF1'], axis=1)
# separate all_data into df_train & df_test
df_train = all_data[:ntrain]
df_test = all_data[ntrain:]
df_train['SalePrice'] = y_train #adding 'SalePrice' column into df_train
#print(df_train.isnull().sum().max()) #just checking that there's no missing data missing

cName_train = df_train.head(0).T # get only column names and transposes(T) row into columns
cName_train.to_csv('../output/column_name_df_train.csv') # column names after handling missing data
df_train.to_csv('../output/df_train.csv')
print("df_train set size after handling missing data(~ID):", df_train.shape)
print("df_test set size after handling missing data(~ID):", df_test.shape)


##################################### 2. Out-liars Handling #############################################
#.......................................2a numerical analyzing.......................................

#checker.numerical_relationship(df_train, 'Total_SF')
#checker.numerical_relationship(df_train, 'BedroomAbvGr')
#checker.numerical_relationship(df_train, 'BedroomAbvGr')
#checker.numerical_relationship(df_train, 'BldgType') #?????????????????????????????????????????????
#checker.numerical_relationship(df_train, 'BsmtFinSF1')
#checker.numerical_relationship(df_train, 'BsmtFinSF2')
#checker.numerical_relationship(df_train, 'BsmtFullBath')
#checker.numerical_relationship(df_train, 'BsmtHalfBath')
#checker.numerical_relationship(df_train, 'BsmtUnfSF')
#checker.numerical_relationship(df_train, 'Condition1')
#checker.numerical_relationship(df_train, 'Condition2')
#checker.numerical_relationship(df_train, 'Electrical')
#checker.numerical_relationship(df_train, 'EnclosedPorch')
#checker.numerical_relationship(df_train, 'Exterior1st')
#checker.numerical_relationship(df_train, 'Exterior2nd')
#checker.numerical_relationship(df_train, 'Fireplaces')
#checker.numerical_relationship(df_train, 'Foundation')
#checker.numerical_relationship(df_train, 'FullBath')
#checker.numerical_relationship(df_train, 'GarageArea')
#checker.numerical_relationship(df_train, 'GarageCars')
#checker.numerical_relationship(df_train, 'GarageType')
#checker.numerical_relationship(df_train, 'GarageYrBlt')

#checker.numerical_relationship(df_train, 'GrLivArea')
drop_index = df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice']<300000)].index
df_train = df_train.drop(drop_index)
#checker.numerical_relationship(df_train, 'GrLivArea')

#checker.numerical_relationship(df_train, 'HalfBath')
#checker.numerical_relationship(df_train, 'Heating')
#checker.numerical_relationship(df_train, 'HouseStyle')
#checker.numerical_relationship(df_train, 'KitchenAbvGr')
#checker.numerical_relationship(df_train, 'LandContour')
#checker.numerical_relationship(df_train, 'LotArea')
#checker.numerical_relationship(df_train, 'LotConfig')
#checker.numerical_relationship(df_train, 'LotFrontage')
#checker.numerical_relationship(df_train, 'LowQualFinSF')
#checker.numerical_relationship(df_train, 'MasVnrArea')
#checker.numerical_relationship(df_train, 'MasVnrType')
#checker.numerical_relationship(df_train, 'MiscFeature')
#checker.numerical_relationship(df_train, 'MiscVal')
#checker.numerical_relationship(df_train, 'MSZoning')
#checker.numerical_relationship(df_train, 'Neighborhood')
#checker.numerical_relationship(df_train, 'OpenPorchSF')
#checker.numerical_relationship(df_train, 'OverallQual')
#checker.numerical_relationship(df_train, 'PoolArea')
#checker.numerical_relationship(df_train, 'RoofMatl')
#checker.numerical_relationship(df_train, 'RoofStyle')
#checker.numerical_relationship(df_train, 'SaleCondition')
#checker.numerical_relationship(df_train, 'SalePrice')
#checker.numerical_relationship(df_train, 'SaleType')
#checker.numerical_relationship(df_train, 'ScreenPorch')
#checker.numerical_relationship(df_train, 'TotalBsmtSF')
#checker.numerical_relationship(df_train, 'TotRmsAbvGrd')
#checker.numerical_relationship(df_train, 'WoodDeckSF')
#checker.numerical_relationship(df_train, 'YearBuilt')
#checker.numerical_relationship(df_train, 'YearRemodAdd')

#.......................................2b categorical analyzing.......................................

#checker.categorical_relationship(df_train, 'Alley')
#checker.categorical_relationship(df_train, 'BsmtCond')
#checker.categorical_relationship(df_train, 'BsmtExposure')
#checker.categorical_relationship(df_train, 'BsmtFinType1')
#checker.categorical_relationship(df_train, 'BsmtFinType2')
#checker.categorical_relationship(df_train, 'BsmtQual')
#checker.categorical_relationship(df_train, 'CentralAir')
#checker.categorical_relationship(df_train, 'ExterCond')
#checker.categorical_relationship(df_train, 'ExterQual')
#checker.categorical_relationship(df_train, 'Fence')
#checker.categorical_relationship(df_train, 'FireplaceQu')
#checker.categorical_relationship(df_train, 'Functional')
#checker.categorical_relationship(df_train, 'GarageCond')
#checker.categorical_relationship(df_train, 'GarageFinish')
#checker.categorical_relationship(df_train, 'GarageQual')
#checker.categorical_relationship(df_train, 'HeatingQC')
#checker.categorical_relationship(df_train, 'KitchenQual')
#checker.categorical_relationship(df_train, 'LandSlope')
#checker.categorical_relationship(df_train, 'LotShape')
#checker.categorical_relationship(df_train, 'MoSold')
#checker.categorical_relationship(df_train, 'MSSubClass')
#checker.categorical_relationship(df_train, 'OverallCond')
#checker.categorical_relationship(df_train, 'PavedDrive')
#checker.categorical_relationship(df_train, 'PoolQC')
#checker.categorical_relationship(df_train, 'Street')
#checker.categorical_relationship(df_train, 'YrSold')

###################################### 3. Normalization handling #########################################

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
#df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
#df_train['HasBsmt'] = 0
#df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
#df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#checker.general_distribution(df_train, 'TotalBsmtSF')
#plt.show()


################################### 3. Homoscedasticity handling #########################################


############################ 4. Converting categorical variable into dummy ###############################
#all_data = pd.get_dummies(all_data) #?????????????????????????????????????????
