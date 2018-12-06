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

# ===================================处理操作详情===================================== #

operation_trn = pd.merge(operation_trn, tag_trn, how='left', on='UID')
operation_test['Tag'] = -1
df = pd.concat([operation_trn, operation_test])


def split_time(v):
    if pd.isna(v):
        return np.nan
    return ' '.join(v.split(':'))

df['time'] = df['time'].apply(lambda x: split_time(x))



def split_version(v):
    if pd.isna(v):
        return np.nan
    return ' '.join(v.split('.'))


df['version'] = df['version'].apply(lambda v: split_version(v))


# df['device2'] 可对型号细分
def split_geo(v):
    if pd.isna(v):
        return np.nan
    return ' '.join(v)
df['geo_code'] = df['geo_code'].apply(lambda x:split_geo(x))

Tag = df.Tag
df = df.drop('Tag',axis=1)
df.insert(0,'Tag',Tag)

df[df.Tag!=-1].to_csv('../input/fm_operation_TRAIN.csv',index=False)
df[df.Tag==-1].drop(['Tag'],axis=1).to_csv('../input/fm_operation_round1.csv',index=False)

# ===================================处理交易详情=====================================#
transaction_trn = pd.merge(transaction_trn, tag_trn, how='left', on='UID')
transaction_test['Tag'] = -1
df = pd.concat([transaction_trn, transaction_test])


df['time'] = df['time'].apply(lambda x: split_time(x))

df['geo_code'] = df['geo_code'].apply(lambda x:split_geo(x))

Tag = df.Tag
df = df.drop('Tag',axis=1)
df.insert(0,'Tag',Tag)

df[df.Tag!=-1].to_csv('../input/fm_transaction_TRAIN.csv',index=False)
df[df.Tag==-1].drop(['Tag'],axis=1).to_csv('../input/fm_transaction_round1.csv',index=False)
