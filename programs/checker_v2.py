import pandas as pd # data processing, CSV file I/O handler(e.g. pd.read_csv)

import matplotlib.pyplot as plt # data manipulation
import seaborn as sns # data presentation
import numpy as np # linear algebra
from scipy.stats import norm #for some statistics
from scipy import stats  # scientific notation handler
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
path = ''
#.........................................data observing.........................................
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
##################################### distribution handling ##############################################

# Checking distribution (histogram and normal probability plot)
def general_distribution(file, cell):
    plt.subplot(1, 2, 1)
    sns.distplot(file[file[cell]>0][cell], fit=norm)
    fig = plt.figure()
    res = stats.probplot(file[file[cell]>0][cell], plot=plt)


# converting distribution in normal (histogram and normal probability plot)
def normalized_distribution(file, cell):
    file[cell] = np.log1p(file[cell])
    general_distribution(file,cell)
