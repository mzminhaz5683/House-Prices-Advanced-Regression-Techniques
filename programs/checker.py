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

#.....Relationship with numerical variables of SalePrice
def scatter(file, var):
    data = pd.concat([file['SalePrice'], file[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    plt.show()










#.....Checking Distribution in general and converting into normal if necessary
# {
# Checking distribution
def general_distribution(file, cell):
    sns.distplot(file[cell], fit=norm)
    fig = plt.figure()
    res = stats.probplot(file[cell], plot=plt)

# converting distribution in normal
def normalized_distribution(file, cell):
    file[cell] = np.log1p(file[cell])
    general_distribution(file,cell)
# }