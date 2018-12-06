import numpy as np
import pandas as pd
import lightgbm as lgb
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import datetime
operation_trn = pd.read_csv('../input/operation_TRAIN.csv')
operation_test = pd.read_csv('../input/operation_round1.csv')
transaction_trn = pd.read_csv('../input/transaction_TRAIN.csv')
transaction_test = pd.read_csv('../input/transaction_round1.csv')
tag_trn = pd.read_csv('../input/tag_TRAIN.csv')
asd
# ===================================处理操作详情===================================== #

operation_trn = pd.merge(operation_trn, tag_trn, how='left', on='UID')
operation_test['Tag'] = -1
df = pd.concat([operation_trn, operation_test])
df['hour'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
del df['time']
label_feature = ['os', 'mode', 'device1', 'device2', 'device_code1', 'device_code2', 'device_code3',
                 'ip1', 'ip2', 'mac1', 'mac2', 'wifi', 'ip1_sub', 'ip2_sub']

for each in label_feature:
    df[each] = pd.factorize(df[each])[0]


def split_version(v, n):
    if pd.isna(v):
        return np.nan
    return int(v.split('.')[n-1])


df['version_1'] = df['version'].apply(lambda v: split_version(v, 1))
df['version_2'] = df['version'].apply(lambda v: split_version(v, 2))
df['version_3'] = df['version'].apply(lambda v: split_version(v, 3))
del df['version']

# df['device2'] 可对型号细分

geo_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'b': 10, 'c':11, 'd':12, 'e':13, 'f':14, 'g':15, 'h':16,  'j':17,
            'k':18, 'm':19, 'n':20, 'p':21, 'q':22, 'r':23, 's':24, 't':25, 'u':26,
            'v':27, 'w':28, 'x':29, 'y':30, 'z':31,}
def split_geo(g, n):
    if pd.isna(g):
        return np.nan
    return geo_dict[g[n-1]]


for i in range(1, 5):
    df['geo_'+str(i)] = df['geo_code'].apply(lambda g: split_geo(g, i))

del df['geo_code']

# ===================================训练操作数据===================================== #
xx_auc = []
xx_submit = []
N = 5
skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    #'max_depth': 3,
    'metric': 'auc',
    'num_leaves': 31, #31
    'learning_rate': 0.02,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'is_unbalance': True,
    'lambda_l1': 0.1
}
X = np.array(df[df.Tag != -1].drop(['Tag', 'UID'], axis=1))
y = np.array(df[df.Tag != -1]['Tag'])
test = np.array(df[df.Tag == -1].drop(['Tag', 'UID'], axis=1))

del df
for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    print('train _K_ flod', k)
    X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=3000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=30,
                    verbose_eval=100,
                    )
    print("eval:%f" % roc_auc_score(y_test, gbm.predict(X_test, num_iteration=gbm.best_iteration)))
    xx_auc.append(gbm.best_score['valid_0']['auc'])
    xx_submit.append(gbm.predict(test, num_iteration=gbm.best_iteration))

print('train_auc:', np.mean(xx_auc))
s = 0
for each in xx_submit:
    s += each
operation_test['Tag'] = list(s/N)
test_index = operation_test.groupby('UID').Tag.mean().index
Tag = operation_test.groupby('UID').Tag.mean().values

test1 = pd.DataFrame(test_index)
test1['Tag'] = Tag
test1.columns = ['UID', 'Tag']
test1['Tag'] = test1['Tag'].apply(lambda x: 1 if x > 1 else x)
test1['Tag'] = test1['Tag'].apply(lambda x: 0 if x < 0 else x)




# ===================================处理交易详情=====================================#
transaction_trn = pd.merge(transaction_trn, tag_trn, how='left', on='UID')
transaction_test['Tag'] = -1
df = pd.concat([transaction_trn, transaction_test])


label_feature = ['channel', 'amt_src1', 'merchant', 'code1', 'code2', 'trans_type1', 'acc_id1',
                  'device_code1', 'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
                    'amt_src2', 'acc_id2', 'acc_id3', 'trans_type2', 'market_code', 'ip1_sub']
for each in label_feature:
    df[each] = pd.factorize(df[each])[0]
df['hour'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
del df['time']
for i in range(1, 5):
    df['geo_'+str(i)] = df['geo_code'].apply(lambda g: split_geo(g, i))
del df['geo_code']

X = np.array(df[df.Tag != -1].drop(['Tag', 'UID'], axis=1))
y = np.array(df[df.Tag != -1]['Tag'])
test = np.array(df[df.Tag == -1].drop(['Tag', 'UID'], axis=1))
xx_auc = []
xx_submit = []
N = 5

for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    print('train _K_ flod', k)
    X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=3000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=30,
                    verbose_eval=100,
                    )
    print(roc_auc_score(y_test, gbm.predict(X_test, num_iteration=gbm.best_iteration)))
    xx_auc.append(gbm.best_score['valid_0']['auc'])
    xx_submit.append(gbm.predict(test, num_iteration=gbm.best_iteration))

print('train_auc:', np.mean(xx_auc))
s = 0
for each in xx_submit:
    s += each
transaction_test['Tag'] = list(s/N)
test_index = transaction_test.groupby('UID').Tag.mean().index
Tag = transaction_test.groupby('UID').Tag.mean().values
test2 = pd.DataFrame(test_index)
test2['Tag'] = Tag
test2.columns = ['UID', 'Tag']
test2['Tag'] = test2['Tag'].apply(lambda x: 1 if x > 1 else x)
test2['Tag'] = test2['Tag'].apply(lambda x: 0 if x < 0 else x)

# ===================================合并预测结果=====================================#
flag = 2

if flag==1:
    # =====================重合的使用test1的结果=====================#0.78
    u2 = set(test2.UID.values) - set(test1.UID.values)
    res = pd.concat([test1, test2[test2.UID.isin(u2)]])
    test_index = res.groupby('UID').Tag.mean().index
    Tag = res.groupby('UID').Tag.mean().values
    res = pd.DataFrame(test_index)
    res['Tag'] = Tag
elif flag==2:
    # =====================重合的使用test2的结果=====================#0.92
    u1 = set(test1.UID.values) - set(test2.UID.values)
    res = pd.concat([test1[test1.UID.isin(u1)], test2])
    test_index = res.groupby('UID').Tag.mean().index
    Tag = res.groupby('UID').Tag.mean().values
    res = pd.DataFrame(test_index)
    res['Tag'] = Tag
else:
    # =====================重合的做平均=====================#0.91
    res = pd.concat([test1, test2])
    test_index = res.groupby('UID').Tag.mean().index
    Tag = res.groupby('UID').Tag.mean().values
    res = pd.DataFrame(test_index)
    res['Tag'] = Tag


#res[['UID','Tag']].to_csv('../submit/result.csv', index = False)