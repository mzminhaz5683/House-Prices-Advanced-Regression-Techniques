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

# Out-liars Handling
# {
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
# }



#.....Normalization handling
# {
# Checking distribution (histogram and normal probability plot)
def general_distribution(file, cell):
    sns.distplot(file[file[cell]>0][cell], fit=norm)
    fig = plt.figure()
    res = stats.probplot(file[file[cell]>0][cell], plot=plt)

# converting distribution in normal (histogram and normal probability plot)
def normalized_distribution(file, cell):
    file[cell] = np.log1p(file[cell])
    general_distribution(file,cell)
# }