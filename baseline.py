# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:19:32 2018

@author: Administrator
"""

import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import random



data_path = '../data/'

train = pd.read_csv('d_train_20180102.csv',encoding='gbk')
test = pd.read_csv('d_test_A_20180102.csv',encoding='gbk')
test2= pd.read_csv('d_test_B_20180128.csv',encoding='gbk')
A_real=pd.read_csv('d_answer_a_20180128.csv',encoding='gbk',header=None)

def make_feat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])

    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
 
    data.fillna(data.median(axis=0),inplace=True)

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat,test_feat



train_feat,test_feat = make_feat(train,test)

predictors = [f for f in test_feat.columns if f not in ['血糖','id','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体']]


few=train_feat[train_feat['血糖']>8.0]
most= train_feat[train_feat['血糖']<=8.0]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score,False)

print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.8,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 80,
    'min_hessian': 1,
    'verbose': -1,
    'bagging_fraction':0.9,
    'bagging_freq':20
}

print('开始CV 5折训练...')
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
#kf1= KFold(len(few),n_folds=10,shuffle=True,random_state=520)

kf2= KFold(len(most),n_folds=5,shuffle=True,random_state=1)

for i, (train_index, test_index) in enumerate(kf):
   
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    #train_feat1=pd.concat([train_feat1,few],axis=0)
    train_feat2 = train_feat.iloc[test_index]
    #train_feat2= pd.concat([train_feat2,few[250:]],axis=0)
    #test_index=train_feat2.index
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'],categorical_feature=['性别'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=500)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:,i] = gbm.predict(test_feat[predictors])
print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'],train_preds)*0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
#submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),header=None,
                  #index=False, float_format='%.4f')


train_preds1 = np.zeros(train_feat.shape[0])
test_preds1 = np.zeros((test_feat.shape[0], 5))
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    #train_feat1=pd.concat([train_feat1,few],axis=0)
    train_feat2 = train_feat.iloc[test_index]
    #train_feat2= pd.concat([train_feat2,few[250:]],axis=0)
    #test_index=train_feat2.index
    GBC=GradientBoostingRegressor(loss='ls',n_estimators=100,max_features=3,max_depth=6,random_state=50)
    GBC.fit(train_feat1[predictors].values,train_feat1['血糖'].values)
    train_preds1[test_index] += GBC.predict(train_feat2[predictors].values)
    test_preds1[:,i] = GBC.predict(test_feat[predictors])
print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'],train_preds1)*0.5))
submission1 = pd.DataFrame({'pred':test_preds1.mean(axis=1)})
#submission1.to_csv(r'sub1{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),header=None,
                  #index=False, float_format='%.4f')

'''
GBC.fit(train_feat[predictors],train_feat['血糖'])
gbc_pred=GBC.predict(test_feat[predictors])
gbc_pred=np.round(gbc_pred,2)
gbc_pred=pd.DataFrame(gbc_pred)
gbc_pred.to_csv('gbc_pred1.csv',index=None,header=None)
'''
submission2=0.6*submission+0.4*submission1
#submission2.to_csv(r'Bsub2{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),header=None,
                  #index=False, float_format='%.4f')

def meserror(pred, df):
    label = df.values.copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score,False)

MSE=meserror(submission2,A_real)
print(MSE)