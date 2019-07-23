import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error , make_scorer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from programs import project
print("______________\nProject_model\n______________")
print('finished stage : import')

X, X_test = project.get_train_test_data()
y = y_train = project.get_train_label()
print('finished stage : input')

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
print('finished stage : KFold')

# model scoring and validation function
def cv_rmse(model_l, X_l=X):
    print('in stage : cv_rmse')
    rmse = np.sqrt(-cross_val_score(model_l, X_l, y,scoring="neg_mean_squared_error",cv=kfolds))
    return rmse

# rmsle scoring function
def rmsle(y_l, y_pred):
    print('in stage : rmsle')
    return np.sqrt(mean_squared_error(y_l, y_pred))

lightgbm = LGBMRegressor(objective='regression',
                                       num_leaves=4, #was 3
                                       learning_rate=0.01,
                                       n_estimators=8000,
                                       max_bin=200,
                                       bagging_fraction=0.75,
                                       bagging_freq=5,
                                       bagging_seed=7,
                                       feature_fraction=0.2, # 'was 0.2'
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
print('finished stage : lightgbm')
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                      max_depth=3, min_child_weight=0,
                                      gamma=0, subsample=0.7,
                                      colsample_bytree=0.7,
                                      objective='reg:linear', nthread=-1,
                                      scale_pos_weight=1, seed=27,
                                      reg_alpha=0.00006)

# setup models hyperparameters using a pipline
# The purpose of the pipeline is to assemble several steps that can be cross-validated together, while setting different parameters.
# This is a range of values that the model considers each time in runs a CV
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
print('finished stage : e_alphas,...alphas2')

# Kernel Ridge Regression : made robust to outliers
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
print('finished stage : ridge')

# LASSO Regression : made robust to outliers
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7,
                    alphas=alphas2,random_state=42, cv=kfolds))
print('finished stage : lasso')


# Elastic Net Regression : made robust to outliers
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7,
                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
print('finished stage : elasticnet')

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, lightgbm),
                                meta_regressor=elasticnet,
                                use_features_in_secondary=True)
print('finished stage : stack_gen')

# store models, scores and prediction values
models = {'Ridge': ridge,
          'Lasso': lasso,
          'ElasticNet': elasticnet,
          'lightgbm': lightgbm,
          'xgboost': xgboost}
print('finished stage : models')

predictions = {}
scores = {}
for name, model in models.items():
    model.fit(X, y)
    predictions[name] = np.expm1(model.predict(X))

    score = cv_rmse(model, X) #function call
    scores[name] = (score.mean(), score.std())


# get the performance of each model on training data(validation set)
print('---- Score with CV_RMSLE-----')
score = cv_rmse(ridge) #function call
print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lasso) #function call
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet) #function call
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lightgbm) #function call
print("lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(xgboost)
print("xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#Fit the training data X, y
print('----START Fit----',datetime.now())
print('Elasticnet')
elastic_model = elasticnet.fit(X, y)
print('Lasso')
lasso_model = lasso.fit(X, y)
print('Ridge')
ridge_model = ridge.fit(X, y)
print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)

print('xgboost')
xgb_model_full_data = xgboost.fit(X, y)

print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

def blend_models_predict(X_l):
    return ((0.25  * elastic_model.predict(X_l)) +
            (0.25 * lasso_model.predict(X_l)) +
            (0.2 * ridge_model.predict(X_l)) +
            (0.10 * lgb_model_full_data.predict(X_l)) +
            (0.1 * xgb_model_full_data.predict(X_l)) +
            (0.2 * stack_gen_model.predict(np.array(X_l))))


print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))

print('Predict submission')
'''
submission = pd.read_csv("../input/sample_submission.csv")
submission.iloc[:,1] = (np.expm1(blend_models_predict(X_test)))

q1 = submission['SalePrice'].quantile(0.0042)
q2 = submission['SalePrice'].quantile(0.99)
# # Quantiles helping us get some extreme values for extremely low or high values
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission.csv", index=False)

# predict
submission = pd.read_csv("../input/sample_submission.csv")
submission.iloc[:,1] = (np.expm1(blend_models_predict(X_test)))
submission.to_csv('submission.csv',index=False)

'''
# my submission
ensemble_predict = blend_models_predict(X_test)
ensemble_predict = np.expm1(ensemble_predict)
sub = pd.DataFrame()
sub['Id'] = project.get_test_ID()
sub['SalePrice'] = ensemble_predict
sub.to_csv('submission.csv',index=False)

print('finished stage : Submission')
