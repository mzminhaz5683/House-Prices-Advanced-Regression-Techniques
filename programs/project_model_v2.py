# coding=utf-8
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))

# import local files & performance parameters
# noinspection PyBroadException
try:
    f_counter = open('../output/contents/model_counter.txt', 'r')
    counter = int(f_counter.read())
    f_counter.close()
except:
    counter = 0

f_counter = open('../output/contents/model_counter.txt', 'w')
counter = str(counter+1)
f_counter.write(counter)
f_counter.close()
######################################### model controller ####################################
from programs import project_v2 as project_analyser
output = 'project_v2'
file_formate = '_m2_'+counter+'.csv'

random_state = 42       # at least 42 (61)
n_estimator = 3000     # at least 3000 (=gbr, +2000 =lightgbm, +460 =xgboost)

# load submission templates
templates_activator = 0

sbmsn_tmplt = {0:[0.5, '0.10375_project_v2_with_4_cTemplate_r49_e3400_m2_72'],
               1:[0.15, '0.10382_project_v2_with_3_cTemplate_r47_e4000_m2_81'],
               2:[0.1, '0.11190_project_v2_with_2_cTemplate_r42_e3000_m2_78'],
               3:[0.05, '0.11292_project_v2_with_3_cTemplate_r51_e3000_m2_67']
              }

###################################################################################################
print("_________________________________________\n")
print('       Project_Model (run stage): {0}               '.format(counter))
print("_________________________________________\n")

print("Model runs with '{0}' template(s).".format(len(sbmsn_tmplt) if templates_activator else 0))
print("For random_state = {0} & n_estimator = {1}\n".format(random_state, n_estimator))

print("_________________________________________\n")
description = '~~~~~~~~~~~~~~~~~~~~~~~ Model file data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'

X, X_test = project_analyser.get_train_test_data()
y = y_train = project_analyser.get_train_label()
print('\n_____________________ Model Algorithms _____________________')
kfolds = KFold(n_splits=10, shuffle=True, random_state=random_state)
description += 'kfolds = KFold(n_splits=10, shuffle=True, random_state={0})\n'.format(random_state)
# rmsle
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# build our model scoring function
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return rmse

# setup models
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas=alphas2,
                              random_state=random_state, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(),
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds,
                                        random_state=random_state, l1_ratio=e_l1ratio))

svr = make_pipeline(RobustScaler(),
                    SVR(C=20, epsilon=0.008, gamma=0.0003, ))

gbr = GradientBoostingRegressor(n_estimators=n_estimator, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10,
                                loss='huber', random_state=random_state)

lightgbm = LGBMRegressor(objective='regression',
                         num_leaves=4,
                         learning_rate=0.01,
                         n_estimators=n_estimator+2000,
                         max_bin=200,
                         bagging_fraction=0.75,
                         bagging_freq=5,
                         bagging_seed=7,
                         feature_fraction=0.2,
                         feature_fraction_seed=7,
                         verbose=-1,
                         )

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=n_estimator+460,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006, random_state=random_state)

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
                                            gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)

print('TEST score on CV')
description += 'TEST score on CV\n_________________\n'

score = cv_rmse(ridge)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )
description += "Kernel Ridge score: {:.4f} ({:.4f}) ~> 0.1138 (0.0154)\n".format(score.mean(), score.std())

score = cv_rmse(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )
description += "Lasso score: {:.4f} ({:.4f}) ~> 0.1138 (0.0148)\n".format(score.mean(), score.std())

score = cv_rmse(elasticnet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )
description += "ElasticNet score: {:.4f} ({:.4f}) ~> 0.1138 (0.0148)\n".format(score.mean(), score.std())

score = cv_rmse(svr)
print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )
description += "SVR score: {:.4f} ({:.4f}) ~> 0.1776 (0.0177)\n".format(score.mean(), score.std())

score = cv_rmse(lightgbm)
print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )
description += "Lightgbm score: {:.4f} ({:.4f}) ~> 0.1142 (0.0164)\n".format(score.mean(), score.std())

score = cv_rmse(gbr)
print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )
description += "GradientBoosting score: {:.4f} ({:.4f}) ~> 0.1135 (0.0151)\n".format(score.mean(), score.std())

score = cv_rmse(xgboost)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )
description += "Xgboost score: {:.4f} ({:.4f}) ~> 0.1154 (0.0166)\n".format(score.mean(), score.std())

print('Start : StackingCVRegressor')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))
print('Start : elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)
print('Start : lasso')
lasso_model_full_data = lasso.fit(X, y)
print('Start : ridge')
ridge_model_full_data = ridge.fit(X, y)
print('Start : svr')
svr_model_full_data = svr.fit(X, y)
print('Start : GradientBoosting')
gbr_model_full_data = gbr.fit(X, y)
print('Start : xgboost')
xgb_model_full_data = xgboost.fit(X, y)
print('Start : lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)


def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) +
            (0.05 * lasso_model_full_data.predict(X)) +
            (0.1 * ridge_model_full_data.predict(X)) +
            (0.1 * svr_model_full_data.predict(X)) +
            (0.1 * gbr_model_full_data.predict(X)) +
            (0.15 * xgb_model_full_data.predict(X)) +
            (0.1 * lgb_model_full_data.predict(X)) +
            (0.3 * stack_gen_model.predict(np.array(X))))

print('RMSLE score on train data')
rmse = rmsle(y, blend_models_predict(X))
print(rmse)
description += 'RMSLE score on train data : {0} ~> 0.0576\n'.format(rmse)

submission = pd.read_csv("../input/sample_submission.csv")

temp = file_formate
file_formate = "_r{0}_e{1}".format(random_state, n_estimator)+temp
######################################## combine results ##########################################
if templates_activator:
    lst = []
    template_weight = 0
    for i in range(0, len(sbmsn_tmplt)):
        template_weight += sbmsn_tmplt[i][0]
        lst.append(pd.read_csv('../output/submission_template/'+sbmsn_tmplt[i][1]+'.csv'))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    submission['SalePrice'] = (     (1.0-template_weight) *
                                    np.expm1(blend_models_predict(X_test))      )

    for i in range(0, len(sbmsn_tmplt)):
        submission['SalePrice'] += (sbmsn_tmplt[i][0] * lst[i].iloc[:,1])

    submission['SalePrice'] = np.floor(submission['SalePrice'])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-

    temp = file_formate
    file_formate = '_with_{0}t'.format(len(sbmsn_tmplt))+temp
else:
    submission['SalePrice'] = np.floor(np.expm1(blend_models_predict(X_test)))
    temp = file_formate
    file_formate = '_with_0t'+temp
######################################## Brutal approach ##########################################
# Brutal approach to deal with predictions close to outer range 
q1 = submission['SalePrice'].quantile(0.0042)
q2 = submission['SalePrice'].quantile(0.99)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
######################################## result ###################################################

submission.to_csv("../output/"+output+file_formate, index=False)
print('________________Stage finished : {0}___________________'.format(counter))
print('Submissin sucessfull saved in output with the name')
print(' ~>> '+output+file_formate)

project_analyser.project_description(description)