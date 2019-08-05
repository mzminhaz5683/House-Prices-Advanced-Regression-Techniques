# coding=utf-8
import pandas as pd # data processing, CSV file I/O handler(e.g. pd.read_csv)

import matplotlib.pyplot as plt # data manipulation
import seaborn as sns # data presentation
import numpy as np # linear algebra
from scipy.stats import norm #for some statistics
from scipy import stats  # scientific notation handler

import warnings
warnings.filterwarnings('ignore')
path = ''
########################################## data observing ########################################
def missing_data(file, save):
    nulls = np.sum(file.isnull())
    nullcols = nulls.loc[(nulls != 0)]
    dtypes = file.dtypes
    dtypes2 = dtypes.loc[(nulls != 0)]

    total = file.isnull().sum().sort_values(ascending=False)
    percent = ((file.isnull().sum()/file.isnull().count()) * 100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent, dtypes2], axis=1, keys=['Total', 'Percent', 'Data Type'])
    if save:
        print(len(nullcols), " missing data, data saves in 'missing_file.csv'")
        missing_data.to_csv(path+'missing_file.csv')
    else:
        print(len(nullcols), " missing data")

def partial(group, relation):
    pd.set_option('max_columns', None)
    var = relation
    var2 = pd.DataFrame([var[i] for i in group]).T
    var2.to_csv(path + 'partial.csv')
    print("partial relation data has been stored in : partial.csv")

##################################### distribution handling ##############################################

# Checking distribution (histogram and normal probability plot)
def general_distribution(file, cell):
    plt.subplot(1, 2, 1)
    sns.distplot(file[file[cell]>0][cell], fit=norm)
    fig = plt.figure()
    res = stats.probplot(file[file[cell]>0][cell], plot=plt)


########################################### Out-liars Handling ############################################

#.....Relationship with numerical variables of SalePrice (Bivariate analysis)
def numerical_relationship(file, var):
    data = pd.concat([file['SalePrice'], file[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    plt.show()

#.....Relationship with categorical features of SalePrice (Bivariate analysis)
def categorical_relationship(file, var):
    data = pd.concat([file['SalePrice'], file[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()

################################## categorical(Ordinal) variables handling ###############################
def data_converter(dic, file, cell):
    file[cell] = file[cell].replace(dic)
    return file[cell].astype(float)
