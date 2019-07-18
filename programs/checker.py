import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O handler(e.g. pd.read_csv)
import seaborn as sns # # data presentation
import matplotlib.pyplot as plt # data manipulation

from datetime import datetime # time handler
from scipy import stats # scientific notation handler
from scipy.stats import norm, skew #for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
#.....Input data files are available in the "../input/" directory.
import os
#print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')

'''
#.....Before the normalisation
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))

#Check the new distribution
sns.distplot(train['SalePrice'], color="b")
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()
'''
def normalization(file, cell):
    #We use the numpy function log1p = log(x+1)
    file[cell] = np.log1p(file[cell])
    sns.set_style("white")
    sns.set_color_codes(palette='deep')
    f, ax = plt.subplots(figsize=(8, 7))
    #Check the new distribution
    sns.distplot(file[cell] , fit=norm, color="b");

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(file[cell])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    ax.xaxis.grid(False)
    ax.set(ylabel="Frequency")
    ax.set(xlabel=cell)
    ax.set(title="Normal distribution")
    sns.despine(trim=True, left=True)

    plt.show()
