import pandas as pd # data processing, CSV file I/O handler(e.g. pd.read_csv)
import matplotlib.pyplot as plt # data manipulation
import seaborn as sns # data presentation
import numpy as np # linear algebra
from scipy.stats import norm #for some statistics
from scipy import stats  # scientific notation handler
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from programs import checker # import local file


##################################### distribution handling ##############################################

# Checking distribution (histogram and normal probability plot)
def general_distribution(file, cell):
    plt.subplot(1, 2, 1)
    sns.distplot(file[file[cell]>0][cell], kde=False, fit=norm)
    fig = plt.figure()
    res = stats.probplot(file[file[cell]>0][cell], plot=plt)


# converting distribution in normal (histogram and normal probability plot)
def normalized_distribution(file, cell):
    file[cell] = np.log1p(file[cell])
    general_distribution(file,cell)
