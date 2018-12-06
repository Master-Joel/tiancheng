import lightgbm as lgb
import pandas as pd
import numpy as np
import datetime
from scipy import sparse
from sklearn.model_selection import train_test_split

# op_train = pd.read_csv('../input/operation_train_new.csv')
tran_train = pd.read_csv('../input/transaction_train_new.csv')
tag_train = pd.read_csv('../input/tag_train_new.csv')
# op_test = pd.read_csv('../input/operation_round1_new.csv')
tran_test = pd.read_csv('../input/transaction_round1_new.csv')
tag_test = pd.read_csv('../input/sample_new.csv')

tran_train = tran_train.merge(tag_train, how='left', on='UID')

tran_train['hour'] = tran_train['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
tran_test['hour'] = tran_test['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
del tran_train['time']
del tran_test['time']

dic = {}
# trans_amt bal
for each in ['channel', 'day', 'trans_amt', 'amt_src1', 'merchant',
             'code1', 'code2', 'trans_type1', 'acc_id1', 'device_code1',
             'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
             'bal', 'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
             'market_code', 'market_type', 'ip1_sub']:
    tmp1 = tran_train.groupby(each).Tag.sum() / tran_train.groupby(each).Tag.count()
    tmp1 = tmp1.reset_index()
    tmp2 = tran_train.groupby(each).Tag.count().reset_index()
    tmp2.columns = [each, 'count']
    tmp1 = tmp1.merge(tmp2,how='left', on=each)
    asset = tmp1[(tmp1.Tag > 0) & (tmp1['count'] > 2)][each].tolist()
    dic[each] = asset



i=1
for fe in dic.keys():
    for v in dic[fe]:
        tmp = tran_train[tran_train[fe] == v].groupby('UID').day.count() / tran_train.groupby('UID').day.count()
        tmp = tmp.fillna(0)
        tmp1 = sparse.coo_matrix(tmp).reshape(-1,1)
        if i == 1:
            train = tmp1
        else:
            train = sparse.hstack((train,tmp1))

        tmp = tran_test[tran_test[fe] == v].groupby('UID').day.count() / tran_test.groupby('UID').day.count()
        tmp = tmp.fillna(0)
        tmp1 = sparse.coo_matrix(tmp).reshape(-1,1)
        if i == 1:
            test = tmp1
            i += 1
        else:
            test = sparse.hstack((test,tmp1))
    print(fe+' done')




def tpr_weight_funtion(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3


params = {
    'boosting_type': 'gbdt',
    'num_leaves':64,
    'reg_alpha':0,
    'reg_lambda':0,
    'max_depth':-1,
    'objective': 'binary',
    'subsample': 0.9,
    'colsample_bytree':0.8,
    'subsample_freq': 1,
    'learning_rate': 0.05,
    'min_child_weight': 4,
    'min_child_samples': 5,
    'min_split_gain': 0
}
N = 5
submit = []
tran_uid = tran_train.UID.unique()
label = tag_train[tag_train.UID.isin(tran_uid)].Tag.values
for k in range(N):
    print('train _K_ flod', k)
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.2)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=3000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=30,
                    verbose_eval=100,
                    )
    print(tpr_weight_funtion(y_test, gbm.predict(X_test, num_iteration=gbm.best_iteration)))

    submit.append(gbm.predict(test, num_iteration=gbm.best_iteration))
asd
s = 0

for each in submit:
    s+=each
s/=N

tag_test['Tag'] = s
tag_test[['UID', 'Tag']].to_csv('../submit/result.csv', index=False)