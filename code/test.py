import lightgbm as lgb
import pandas as pd
import numpy as np
import datetime
from scipy import sparse
from sklearn.model_selection import train_test_split

op_train = pd.read_csv('../input/operation_train_new.csv')
tran_train = pd.read_csv('../input/transaction_train_new.csv')
tag_train = pd.read_csv('../input/tag_train_new.csv')
op_test = pd.read_csv('../input/operation_round1_new.csv')
tran_test = pd.read_csv('../input/transaction_round1_new.csv')
tag_test = pd.read_csv('../input/sample_new.csv')

tran_train = tran_train.merge(tag_train, how='left', on='UID')
op_train = op_train.merge(tag_train, how='left', on='UID')
tran_train['hour'] = tran_train['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
tran_test['hour'] = tran_test['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
del tran_train['time']
del tran_test['time']

dic = {}
# trans_amt bal
for each in [ 'channel', 'trans_amt', 'amt_src1', 'merchant', 'code1',
       'code2', 'trans_type1', 'acc_id1', 'device_code1', 'device_code2',
       'device_code3', 'device1', 'device2', 'mac1', 'ip1', 'amt_src2',
       'acc_id2', 'acc_id3', 'geo_code', 'trans_type2', 'market_code',
       'market_type', 'ip1_sub']:

    tmp = tran_train[tran_train.Tag == 1].groupby(each).UID.nunique() / tran_train.groupby(each).UID.nunique()
    tmp = tmp.reset_index()
    tmp.columns = [each, 'UID_rate']
    tmp1 = tran_train.groupby(each).UID.nunique().reset_index()
    tmp1.columns = [each, 'UID_count']
    tmp = tmp.merge(tmp1, how='left', on=each)
    dic[each] = tmp[(tmp['UID_rate'] > 0.8) & (tmp.UID_count > 40)][each].tolist()

# dic = {}
# for each in [ 'mode', 'success', 'os', 'version', 'device1',
#        'device2', 'device_code1', 'device_code2', 'device_code3', 'mac1',
#        'mac2', 'ip1', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub']:
#
#     tmp = op_train[op_train.Tag == 1].groupby(each).UID.nunique() / op_train.groupby(each).UID.nunique()
#     tmp = tmp.reset_index()
#     tmp.columns = [each, 'UID_rate']
#     tmp1 = op_train.groupby(each).UID.nunique().reset_index()
#     tmp1.columns = [each, 'UID_count']
#     tmp = tmp.merge(tmp1, how='left', on=each)
#     dic[each] = tmp[(tmp['UID_rate'] > 0.8) & (tmp.UID_count > 40)][each].tolist()
#

i=1
for fe in dic.keys():
    for v in dic[fe]:
        tmp = tran_train[tran_train[fe] == v].groupby('UID').day.count() / tran_train.groupby('UID').day.count()
        tmp = tmp.fillna(0).apply(lambda x: 1 if x > 0 else 0)
        tmp = tmp.values.reshape(-1,1).astype(np.float32)
        tmp1 = sparse.coo_matrix(tmp)
        if i == 1:
            train = tmp1
        else:
            train = sparse.hstack((train, tmp1))

        tmp = tran_test[tran_test[fe] == v].groupby('UID').day.count() / tran_test.groupby('UID').day.count()
        tmp = tmp.fillna(0).apply(lambda x: 1 if x > 0 else 0)
        tmp = tmp.values.reshape(-1,1).astype(np.float32)
        tmp1 = sparse.coo_matrix(tmp)
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
    'metric':'auc',
    'num_leaves':64,
    'reg_alpha':0,
    'reg_lambda':0,
    'max_depth':-1,
    'objective': 'binary',
    'subsample': 0.9,
    'colsample_bytree':0.8,
    'subsample_freq': 1,
    'learning_rate': 0.01,
    'min_child_weight': 4,
    'min_child_samples': 5,
    'min_split_gain': 0,
    'is_unbalance': True
}
N = 1
submit = []
tran_uid = tran_train.UID.unique()
test_uid = tran_test.UID.unique()
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
                    verbose_eval=100
                    )
    print(tpr_weight_funtion(y_test, gbm.predict(X_test, num_iteration=gbm.best_iteration)))

    submit.append(gbm.predict(test, num_iteration=gbm.best_iteration))

s = 0

for each in submit:
    s+=each
s/=N

test_uid = pd.DataFrame(test_uid, columns=['UID'])
test_uid['Tag'] = s
del tag_test['Tag']
tag_test = tag_test.merge(test_uid,how='left', on='UID')
tag_test.fillna(0.5,inplace=True)

a = pd.read_csv('../submit/66826.csv')
a['Tag'] = a['Tag'].apply(lambda x: 1 if x > 0.5 else 0)
print(tpr_weight_funtion(a.Tag.values, tag_test.Tag.values))
#tag_test[['UID', 'Tag']].to_csv('../submit/result.csv', index=False)