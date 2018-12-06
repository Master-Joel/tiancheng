
import pandas as pd
import datetime
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import gc
import pickle
# 如果用embedding 每个字段用户每个字段所有记录转成向量累加再平均，可以表示总体趋势

operation_trn = pd.read_csv('../input/operation_TRAIN_new.csv')
operation_test = pd.read_csv('../input/operation_round1_new.csv')
transaction_trn = pd.read_csv('../input/transaction_TRAIN_new.csv')
transaction_test = pd.read_csv('../input/transaction_round1_new.csv')
tag_trn = pd.read_csv('../input/tag_TRAIN_new.csv')
tag_test = pd.read_csv('../input/submission_sample.csv')

load_file=open("Uni.bin","rb")
val_UID=pickle.load(load_file)
load_file.close()


# operation_trn
# UID              29728
# day                 30
# mode                89
# success              2
# time             80670
# os                   7
# version             38
# device1           2421
# device2           1652
# device_code1     26184
# device_code2     32043
# device_code3      6285
# mac1             11276
# mac2             20249
# ip1             141594
# ip2              19989
# wifi             27712
# geo_code          4056
# ip1_sub          35290
# ip2_sub          10449

# transaction_trn
# UID             30542
# channel             5
# day                30
# time            60075
# trans_amt       11225
# amt_src1           28
# merchant        19766
# code1            6101
# code2             668
# trans_type1        15
# acc_id1         27630
# device_code1    28601
# device_code2    28234
# device_code3     7221
# device1          2566
# device2          1697
# mac1            18545
# ip1             76902
# bal             12307
# amt_src2          115
# acc_id2          9887
# acc_id3         13713
# geo_code         3628
# trans_type2         4
# market_code       430
# market_type         2
# ip1_sub         23660

# 对于类别特征count 和 nunique只计算了多少个有值，和这些值有多少类，但是缺少各类的多少/比例
# 没对day这个相对时间做处理，其实测试集晚于训练集，可以训练1-30，测试每个都加上30
# geo_code 只计算了计数和类别数，其实可以解析出距离，最大最小均值方差
# op 中device2可做手机品牌划分，version可以做大版本划分
# time可提取小时，分钟, 额外加分桶早中晚信息？   分桶和次数也要挂钩，某人偏向某个时间段操作（通过小时分钟的均值方差等特性）

op = pd.concat([operation_trn, operation_test])
tran = pd.concat([transaction_trn, transaction_test])

label = pd.concat([tag_trn, tag_test])
num = tag_trn.shape[0]
num_op = operation_trn.shape[0]
num_tran = transaction_trn.shape[0]

gc.collect()
op['hour'] = op['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
tran['hour'] = tran['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)

op['minute'] = op['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').minute)
tran['minute'] = tran['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').minute)
def getPer(x):
    if x >= 0 and x <= 6:
        return 1
    elif x >= 7 and x <= 12:
        return 2
    elif x >= 13 and x <= 18:
        return 3
    elif x >= 19 and x <= 23:
        return 4

op['hour_period'] = op['hour'].apply(lambda x: getPer(x))
tran['hour_period'] = tran['hour'].apply(lambda x: getPer(x))
del op['time']
del tran['time']


# ------------------------------------------------------------------------------------------------------

# device2 合并为品牌
op['device2'] = op['device2'].fillna('Nan').apply(
    lambda x: x.split(' ')[0].split('-')[0].split('_')[0])
tran['device2'] = tran['device2'].fillna('Nan').apply(
    lambda x: x.split(' ')[0].split('-')[0].split('_')[0])
op['device2'] = op['device2'].apply(lambda x: 'IPHONE' if x.startswith('IPHONE') else x)
op['device2'] = op['device2'].apply(lambda x: 'OPPO' if x.startswith('OPPO') else x)
op['device2'] = op['device2'].apply(lambda x: 'VIVO' if x.startswith('VIVO') else x)
op['device2'] = op['device2'].apply(lambda x: 'GN' if x.startswith('GN') else x)
op['device2'] = op['device2'].apply(lambda x: 'MI' if (x.startswith('MI') or x=='XIAOMI') else x)
op['device2'] = op['device2'].apply(lambda x: 'MEIZU' if (x=='M' or x.startswith('MX') or (x[0] == 'M' and x[1].isdigit())) else x)

tran['device2'] = tran['device2'].apply(lambda x: 'IPHONE' if x.startswith('IPHONE') else x)
tran['device2'] = tran['device2'].apply(lambda x: 'OPPO' if x.startswith('OPPO') else x)
tran['device2'] = tran['device2'].apply(lambda x: 'VIVO' if x.startswith('VIVO') else x)
tran['device2'] = tran['device2'].apply(lambda x: 'GN' if x.startswith('GN') else x)
tran['device2'] = tran['device2'].apply(lambda x: 'MI' if (x.startswith('MI') or x=='XIAOMI') else x)
tran['device2'] = tran['device2'].apply(lambda x: 'MEIZU' if (x=='M' or x.startswith('MX') or (x[0] == 'M' and x[1].isdigit()))  else x)




def split_version(v, n):
    if pd.isna(v):
        return np.nan
    return int(v.split('.')[n-1])

op['version_1'] = op['version'].apply(lambda v: split_version(v, 1))

# op1 = op.iloc[:num_op]
# op2 = op.iloc[num_op:]
# tran1 = tran.iloc[:num_tran]
# tran2 = tran.iloc[num_tran:]
# =============================================操作特征提取========================================#
# 每个UID 操作次数
tmp = op.groupby('UID').day.count().reset_index()
tmp.columns = ['UID', 'op_count']
label = label.merge(tmp, on='UID', how='left')

l = op.columns.tolist()
l.remove('UID')
l.remove('version_1')
for each in l:
    # 每个UID 操作中各个字段出现种类数
    tmp = op.groupby('UID')[each].nunique().reset_index()
    tmp.columns = ['UID', 'op_'+each+'_type']
    label = label.merge(tmp, on='UID', how='left')


# 操作中下列字段各种取值比例 （贝叶斯平滑） 仅仅是每个UID下各个特征取值比例
for each in ['mode', 'success', 'os', 'version_1']:
    tmp = op.groupby(['UID', each]).mode.count() / op.groupby('UID').mode.count()
    tmp = tmp.unstack().reset_index()
    label = label.merge(tmp, on='UID', how='left')


# 操作中下列字段出现频率最高的（1）个值, 去掉了ip1因为取值太多会报错
# for each in ['day', 'hour', 'device1', 'device2', 'device_code1',
#              'device_code2', 'device_code3', 'mac1',
#              'mac2', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub']:
#
#     tmp1 = op1.groupby('UID')[each].value_counts().unstack().idxmax(axis=1).reset_index()
#     tmp1.columns = ['UID', 'op_' + each + '_max']
#     tmp2 = op2.groupby('UID')[each].value_counts().unstack().idxmax(axis=1).reset_index()
#     tmp2.columns = ['UID', 'op_' + each + '_max']
#     tmp = pd.concat([tmp1, tmp2])
#     label = label.merge(tmp, on='UID', how='left')


# 操作中下列字段分布情况

# 操作中 时间交叉特征
# 所有每天的交易hour 最大-最小差值 的最大值
tmp = op.groupby(['UID', 'day']).hour.max() - op.groupby(['UID', 'day']).hour.min()
tmp = tmp.max(level='UID').reset_index()
label = label.merge(tmp, on='UID', how='left')
# 每天平均，最大，最小操作数
tmp = op.groupby(['UID', 'day']).hour.count().max(level='UID').reset_index()
label = label.merge(tmp, on='UID', how='left')
tmp = op.groupby(['UID', 'day']).hour.count().min(level='UID').reset_index()
label = label.merge(tmp, on='UID', how='left')
label['op_count_mean'] = label['op_count'] / label['op_day_type']

# 操作中 每个UID 下列特征缺失数 和 day交叉
for each in ['success', 'version', 'device1', 'device2', 'device_code1', 'device_code2', 'device_code3', 'mac1', 'mac2',
             'ip1', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub']:
    tmp = op.groupby(['UID', 'day']).day.count() - op.groupby(['UID', 'day'])[each].count()

    # 总共缺失数
    tmp1 = tmp.sum(level='UID').reset_index()
    tmp1.columns = ['UID', 'op_' + each + '_NanSum']
    label = label.merge(tmp1, on='UID', how='left')
    # 平均每天缺失数
    label['op_' + each + '_Nan_day_mean'] = label['op_' + each + '_NanSum'] / label['op_day_type']
    # 缺失率
    label['op_' + each + '_Nan_rate'] = label['op_' + each + '_NanSum'] / label['op_count']

    # 每天 最大最小缺失数
    tmp1 = tmp.max(level='UID').reset_index()
    tmp1.columns = ['UID', 'op_' + each + '_Nan_day_max']
    label = label.merge(tmp1, on='UID', how='left')
    tmp1 = tmp.min(level='UID').reset_index()
    tmp1.columns = ['UID', 'op_' + each + '_Nan_day_min']
    label = label.merge(tmp1, on='UID', how='left')



# =============================================交易特征提取========================================#
# 每个UID 交易次数
tmp = tran.groupby('UID').day.count().reset_index()
tmp.columns = ['UID', 'tran_count']
label = label.merge(tmp, on='UID', how='left')


l = tran.columns.tolist()
l.remove('UID')
l.remove('trans_amt')
l.remove('bal')
for each in l:
    # 每个UID 交易中各个字段出现种类数
    tmp = tran.groupby('UID')[each].nunique().reset_index()
    tmp.columns = ['UID', 'tran_'+each+'_type']
    label = label.merge(tmp, on='UID', how='left')

# 交易中数值特征
tran['money1'] = tran['trans_amt'] + tran['bal']
tran['money2'] = tran['bal'] - tran['trans_amt']
for each in ['trans_amt', 'bal', 'money1', 'money2']:
    tmp = tran.groupby('UID')[each].max().reset_index()
    tmp.columns = ['UID', each+'_max']
    label = label.merge(tmp, on='UID', how='left')

    tmp = tran.groupby('UID')[each].min().reset_index()
    tmp.columns = ['UID', each + '_min']
    label = label.merge(tmp, on='UID', how='left')

    tmp = tran.groupby('UID')[each].sum().reset_index()
    tmp.columns = ['UID', each + '_sum']
    label = label.merge(tmp, on='UID', how='left')

    tmp = tran.groupby('UID')[each].mean().reset_index()
    tmp.columns = ['UID', each + '_mean']
    label = label.merge(tmp, on='UID', how='left')

    tmp = tran.groupby('UID')[each].median().reset_index()
    tmp.columns = ['UID', each + '_median']
    label = label.merge(tmp, on='UID', how='left')

    tmp = tran.groupby('UID')[each].std().reset_index()
    tmp.columns = ['UID', each + '_std']
    label = label.merge(tmp, on='UID', how='left')

    tmp = tran.groupby('UID')[each].skew().reset_index()
    tmp.columns = ['UID', each + '_skew']
    label = label.merge(tmp, on='UID', how='left')
    label[ each + 'len'] = label[each + '_max'] - label[each + '_min']

# 交易中下列字段各种取值比例    （贝叶斯平滑） 仅仅是每个UID下各个特征取值比例，可以加上channel下。。。
for each in ['channel', 'amt_src1', 'trans_type1', 'trans_type2', 'market_type']:
    tmp = tran.groupby(['UID', each]).day.count() / tran.groupby('UID').day.count()
    tmp = tmp.unstack().reset_index()
    label = label.merge(tmp, on='UID', how='left')



# 交易中下列字段出现频率最高的（1）个值, 去掉了ip1因为取值太多会报错
# for each in ['day', 'hour', 'merchant',
#              'code1', 'code2', 'acc_id1', 'device_code1',
#              'device_code2', 'device_code3', 'device1', 'device2', 'mac1',
#              'amt_src2', 'acc_id2', 'acc_id3', 'geo_code',
#              'market_code', 'ip1_sub']:
#     tmp1 = tran1.groupby('UID')[each].value_counts().unstack().idxmax(axis=1).reset_index()
#     tmp1.columns = ['UID', 'tran_' + each + '_max']
#     tmp2 = tran2.groupby('UID')[each].value_counts().unstack().idxmax(axis=1).reset_index()
#     tmp2.columns = ['UID', 'tran_' + each + '_max']
#     tmp = pd.concat([tmp1, tmp2])
#     label = label.merge(tmp, on='UID', how='left')


# 交易中 时间交叉特征
# 交易中 所有每天的交易hour 最大-最小差值 的最大值
tmp = tran.groupby(['UID', 'day']).hour.max() - tran.groupby(['UID', 'day']).hour.min()
tmp = tmp.max(level='UID').reset_index()
label = label.merge(tmp, on='UID', how='left')
# 每天平均，最大，最小交易数
tmp = tran.groupby(['UID', 'day']).hour.count().max(level='UID').reset_index()
label = label.merge(tmp, on='UID', how='left')
tmp = tran.groupby(['UID', 'day']).hour.count().min(level='UID').reset_index()
label = label.merge(tmp, on='UID', how='left')
label['tran_count_mean'] = label['tran_count'] / label['tran_day_type']

# 交易中 每个UID 下列特征缺失数 和 day交叉
for each in ['code1', 'code2', 'acc_id1', 'device_code1', 'device_code2', 'device_code3', 'device1',
             'mac1', 'ip1', 'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
             'market_code']:
    tmp = tran.groupby(['UID', 'day']).day.count() - tran.groupby(['UID', 'day'])[each].count()

    # 总共缺失数
    tmp1 = tmp.sum(level='UID').reset_index()
    tmp1.columns = ['UID', 'tran_' + each + '_NanSum']
    label = label.merge(tmp1, on='UID', how='left')
    # 平均每天缺失数
    label['tran_' + each + '_Nan_day_mean'] = label['tran_' + each + '_NanSum'] / label['tran_day_type']
    # 缺失率
    label['tran_' + each + '_Nan_rate'] = label['tran_' + each + '_NanSum'] / label['tran_count']

    # 每天 最大最小缺失数
    tmp1 = tmp.max(level='UID').reset_index()
    tmp1.columns = ['UID', 'tran_' + each + '_Nan_day_max']
    label = label.merge(tmp1, on='UID', how='left')
    tmp1 = tmp.min(level='UID').reset_index()
    tmp1.columns = ['UID', 'tran_' + each + '_Nan_day_min']
    label = label.merge(tmp1, on='UID', how='left')

# 每天给几个/种商户交易  最大最小均值之类
tmp = tran.groupby(['UID', 'day']).merchant.count()
tmp1 = tmp.max(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_count_max']
label = label.merge(tmp1, on='UID', how='left')
tmp1 = tmp.mean(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_count_mean']
label = label.merge(tmp1, on='UID', how='left')
tmp1 = tmp.min(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_count_min']
label = label.merge(tmp1, on='UID', how='left')
tmp1 = tmp.std(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_count_min']
label = label.merge(tmp1, on='UID', how='left')
tmp1 = tmp.median(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_count_min']
label = label.merge(tmp1, on='UID', how='left')
tmp1 = tmp.skew(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_count_min']
label = label.merge(tmp1, on='UID', how='left')

tmp = tran.groupby(['UID', 'day']).merchant.nunique()
tmp1 = tmp.max(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_type_max']
label = label.merge(tmp1, on='UID', how='left')
tmp1 = tmp.mean(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_type_mean']
label = label.merge(tmp1, on='UID', how='left')
tmp1 = tmp.min(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_type_min']
label = label.merge(tmp1, on='UID', how='left')
tmp1 = tmp.std(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_type_min']
label = label.merge(tmp1, on='UID', how='left')
tmp1 = tmp.median(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_type_min']
label = label.merge(tmp1, on='UID', how='left')
tmp1 = tmp.skew(level='UID').reset_index()
tmp1.columns = ['UID', 'daily_merchant_type_min']
label = label.merge(tmp1, on='UID', how='left')
# =============================================操作交易特征交叉========================================#
# 用户在连续一段时间内的记录视为一次会话
# 会话次数
op['timestamp'] = op['day']*1440 + op['hour']*60 + op['minute']
tran['timestamp'] = tran['day']*1440 + tran['hour']*60 + tran['minute']
tmp1 = op[['UID', 'day', 'timestamp','device1', 'device2', 'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 'ip1_sub', 'mac1']]
tmp1['type'] = 0
tmp2 = tran[['UID', 'day', 'timestamp','device1', 'device2', 'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 'ip1_sub', 'mac1']]
tmp2['type'] = 1
op_tran = pd.concat([tmp1, tmp2])

op_tran = op_tran.sort_values('timestamp',kind='mergesort')
op_tran = op_tran.sort_values('UID',kind='mergesort')

from collections import defaultdict
action_count = defaultdict(int)
def count_action(action_count, UID, timestamp):
    if action_count[UID] == 0:
        action_count[UID] = [timestamp,1]
        return 1
    else:
        if abs(action_count[UID][0] - timestamp) <= 10:
            action_count[UID] = [timestamp, action_count[UID][1]]
        else:
            action_count[UID] = [timestamp, action_count[UID][1]+1]
        return action_count[UID][1]

op_tran['period'] = op_tran.apply(lambda x: count_action(action_count, x['UID'], x['timestamp']), axis=1)
del action_count

op = op.sort_values('timestamp',kind='mergesort')
op = op.sort_values('UID',kind='mergesort')
tran = tran.sort_values('timestamp',kind='mergesort')
tran = tran.sort_values('UID',kind='mergesort')

op['period'] = op_tran[op_tran['type'] == 0]['period']
tran['period'] = op_tran[op_tran['type'] == 1]['period']

tmp = op_tran.groupby('UID')['period'].max().reset_index()
label = label.merge(tmp, how='left', on='UID')
# 交易和操作  下列特征在每天/每个时段/会话 每个取值的点击率（sum/count）https://zhuanlan.zhihu.com/p/47807544


# 历史点击率
# for feat_1 in ['device1', 'device2', 'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 'ip1_sub', 'mac1']:
#     res = pd.DataFrame()
#     temp = op_tran[[feat_1, 'day', 'type']]
#     for day in range(1, 31):
#         if day == 1:
#             count = temp.groupby([feat_1]).apply(
#                 lambda x: x['type'][(x['day'] <= day).values].count()).reset_index(name=feat_1 + '_all')
#             count1 = temp.groupby([feat_1]).apply(
#                 lambda x: x['type'][(x['day'] <= day).values].sum()).reset_index(name=feat_1 + '_1')
#         else:
#             count = temp.groupby([feat_1]).apply(
#                 lambda x: x['type'][(x['day'] < day).values].count()).reset_index(name=feat_1 + '_all')
#             count1 = temp.groupby([feat_1]).apply(
#                 lambda x: x['type'][(x['day'] < day).values].sum()).reset_index(name=feat_1 + '_1')
#         count[feat_1 + '_1'] = count1[feat_1 + '_1']
#         count.fillna(value=0, inplace=True)
#         count[feat_1 + '_rate'] = round(count[feat_1 + '_1'] / count[feat_1 + '_all'], 5)
#         count['day'] = day
#         count.drop([feat_1 + '_all', feat_1 + '_1'], axis=1, inplace=True)
#         count.fillna(value=0, inplace=True)
#         res = res.append(count, ignore_index=True)
#     print(feat_1, ' over')
#     op_tran = pd.merge(op_tran, res, how='left', on=[feat_1, 'day'])
#
#     tmp = op_tran.groupby('UID')[feat_1 + '_rate']
#     tmp1 = tmp.max().reset_index()
#     tmp1.columns = ['UID',feat_1 + '_rate_max']
#     label = label.merge(tmp1, how='left', on='UID')
#     tmp1 = tmp.min().reset_index()
#     tmp1.columns = ['UID', feat_1 + '_rate_min']
#     label = label.merge(tmp1, how='left', on='UID')
#     tmp1 = tmp.mean().reset_index()
#     tmp1.columns = ['UID', feat_1 + '_rate_mean']
#     label = label.merge(tmp1, how='left', on='UID')
#     tmp1= tmp.agg(lambda x: np.mean(pd.Series.mode(x))).reset_index()
#     tmp1.columns = ['UID', feat_1 + '_rate_mode']
#     label = label.merge(tmp1, how='left', on='UID')

# 每次会话操作、交易、操作和交易的次数   max，min之类
tmp = op_tran[op_tran['type'] == 0].groupby(['UID', 'period']).day.count()
tmp1 = tmp.max(level='UID').reset_index()
tmp1.columns = ['UID', 'period_op_max']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.min(level='UID').reset_index()
tmp1.columns = ['UID', 'period_op_min']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.mean(level='UID').reset_index()
tmp1.columns = ['UID', 'period_op_mean']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.median(level='UID').reset_index()
tmp1.columns = ['UID', 'period_op_median']
label = label.merge(tmp1, how='left', on='UID')


tmp = op_tran[op_tran['type'] == 1].groupby(['UID', 'period']).day.count()
tmp1 = tmp.max(level='UID').reset_index()
tmp1.columns = ['UID', 'period_tran_max']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.min(level='UID').reset_index()
tmp1.columns = ['UID', 'period_tran_min']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.mean(level='UID').reset_index()
tmp1.columns = ['UID', 'period_tran_mean']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.median(level='UID').reset_index()
tmp1.columns = ['UID', 'period_tran_median']
label = label.merge(tmp1, how='left', on='UID')

tmp = op_tran[op_tran['type'] == 1].groupby(['UID', 'period']).day.count() / op_tran[op_tran['type'] == 0].groupby(['UID', 'period']).day.count()
tmp1 = tmp.max(level='UID').reset_index()
tmp1.columns = ['UID', 'period_tran_max']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.min(level='UID').reset_index()
tmp1.columns = ['UID', 'period_tran_min']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.mean(level='UID').reset_index()
tmp1.columns = ['UID', 'period_tran_mean']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.median(level='UID').reset_index()
tmp1.columns = ['UID', 'period_tran_median']
label = label.merge(tmp1, how='left', on='UID')

tmp = op_tran.groupby(['UID', 'period']).day.count()
tmp1 = tmp.max(level='UID').reset_index()
tmp1.columns = ['UID', 'period_ot_max']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.min(level='UID').reset_index()
tmp1.columns = ['UID', 'period_ot_min']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.mean(level='UID').reset_index()
tmp1.columns = ['UID', 'period_ot_mean']
label = label.merge(tmp1, how='left', on='UID')
tmp1 = tmp.median(level='UID').reset_index()
tmp1.columns = ['UID', 'period_ot_median']
label = label.merge(tmp1, how='left', on='UID')



# 交易/操作次数
label['op_tran_rate'] = label['tran_count'] / label['op_count']

# 操作和交易，每天/会话中的ip1，ip1_sub变化次数 geo_code改变，第1，2，3，4位发生改变

for fe in ['device1', 'device2', 'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 'ip1_sub', 'mac1']:
    tmp = op_tran.groupby(['UID', 'period'])[fe].nunique()
    tmp1 = tmp.max(level='UID').reset_index()
    tmp1.columns = ['UID', 'period_ot_' + fe + '_max']
    label = label.merge(tmp1, how='left', on='UID')
    tmp1 = tmp.min(level='UID').reset_index()
    tmp1.columns = ['UID', 'period_ot_' + fe + '_min']
    label = label.merge(tmp1, how='left', on='UID')
    tmp1 = tmp.mean(level='UID').reset_index()
    tmp1.columns = ['UID', 'period_ot_' + fe + '_mean']
    label = label.merge(tmp1, how='left', on='UID')
    tmp1 = tmp.median(level='UID').reset_index()
    tmp1.columns = ['UID', 'period_ot_' + fe + '_median']
    label = label.merge(tmp1, how='left', on='UID')

    tmp = op_tran.groupby(['UID', 'day'])[fe].nunique()
    tmp1 = tmp.max(level='UID').reset_index()
    tmp1.columns = ['UID', 'day_ot_' + fe + '_max']
    label = label.merge(tmp1, how='left', on='UID')
    tmp1 = tmp.min(level='UID').reset_index()
    tmp1.columns = ['UID', 'day_ot_' + fe + '_min']
    label = label.merge(tmp1, how='left', on='UID')
    tmp1 = tmp.mean(level='UID').reset_index()
    tmp1.columns = ['UID', 'day_ot_' + fe + '_mean']
    label = label.merge(tmp1, how='left', on='UID')
    tmp1 = tmp.median(level='UID').reset_index()
    tmp1.columns = ['UID', 'day_ot_' + fe + '_median']
    label = label.merge(tmp1, how='left', on='UID')


# 用户使用手机型号 device2 各品牌比例（需要各型号统一品牌），还有其他特征在黑白显著差异取值上比例
# tmp1 = op_tran.groupby('UID').day.count()
# for each in ['Nan', 'IPHONE', 'MI', 'SM', 'MEIZU', 'ZTE', 'BLN', 'HM','BND', 'FRD', 'COOLPAD',
#  'STF', 'HISENSE', 'CAM', 'PRO', 'MYA', 'DIG', 'KINGSUN', 'DLI', 'LENOVO', 'SCL', 'KNT', 'CHM', 'NEM', 'ONEPLUS', 'CUN',
# 'JMM','GT', 'IPAD', 'LON', 'PE', 'ATH', 'BF', 'HTC', 'TCL', 'VIRTUAL', 'SOP','GM',
#  'SCH', 'LETV', 'NX563J', 'H60', 'E6782', 'NX511J', 'R7005', 'WP', 'K', 'LA', 'C1330', 'NX529J', 'OWWO',
#  'IPOD', 'MP1602', 'SAMSUNG', 'R821T', '15', 'HLTE200T', 'SM919', 'HT', 'HS', 'HLJ', 'NX523J', 'X800', 'U20', 'NUOFEI',
#  'ONE', 'R831S', 'XT1581', 'YQ601', 'DRA', 'F106L', 'X900', 'TETC', 'P6', 'F103S', 'HONGMI', 'DAZEN', 'U8860','MP1603', 'G620',
# 'CLIQ', 'G621', 'LA2', 'BTV', 'X800+', 'YLT', 'T1', 'C03', 'KONKA', '2014011', 'KEJIAN', 'X9007', 'A51KC', '3005', 'R827T',
#  '2014811', 'XT615', 'LEX720', '1509', 'G0121', 'N958ST', 'C8813', 'R5', 'A11', 'SA', '2014821','IPOD7,1', 'U8150', 'MLLED',
# 'G0128', 'CLOAKME', 'X909T','A51', 'SM801', 'BEST', 'GODONIE', 'U8220', '红米', 'Χ', 'Z10', 'PH', 'WA1', 'C105','SM901',
# 'YQ607','3007','SGH','EG750','GALAXY','MP1709','AR9','AUX','R6','E3T', 'V188', '20161220', 'G510','OX1','Y83','G0111',
# 'A1001', 'D5103','IPAD4,3','魅蓝3','HEDY','R8200','LD800','8H','SONY','W806','红米NOTE4X','C8812','X600','DOING','R830']:
#     tmp = op_tran[op_tran.device2 == each].groupby('UID').day.count() / tmp1
#     tmp = tmp.fillna(0).reset_index()
#     tmp.columns = ['UID', 'device2_'+each]
#     label = label.merge(tmp, how='left', on='UID')

# 操作和交易中数值特征    貌似分开算分数好

op_tran = pd.concat([op[['UID','day','hour','hour_period','period']], tran[['UID','day','hour','hour_period','period']]])
for each in ['hour_period', 'day', 'hour', 'period']:
    tmp = op_tran.groupby('UID')[each].max().reset_index()
    tmp.columns = ['UID', each + '_max']
    label = label.merge(tmp, on='UID', how='left')

    tmp = op_tran.groupby('UID')[each].min().reset_index()
    tmp.columns = ['UID',  each + '_min']
    label = label.merge(tmp, on='UID', how='left')

    tmp = op_tran.groupby('UID')[each].sum().reset_index()
    tmp.columns = ['UID',  each + '_sum']
    label = label.merge(tmp, on='UID', how='left')

    tmp = op_tran.groupby('UID')[each].mean().reset_index()
    tmp.columns = ['UID',  each + '_mean']
    label = label.merge(tmp, on='UID', how='left')

    tmp = op_tran.groupby('UID')[each].median().reset_index()
    tmp.columns = ['UID', each + '_median']
    label = label.merge(tmp, on='UID', how='left')

    tmp = op_tran.groupby('UID')[each].std().reset_index()
    tmp.columns = ['UID',  each + '_std']
    label = label.merge(tmp, on='UID', how='left')

    tmp = op_tran.groupby('UID')[each].skew().reset_index()
    tmp.columns = ['UID', each + '_skew']
    label = label.merge(tmp, on='UID', how='left')

    label[each + '_len'] = label[ each + '_max'] - label[ each + '_min']


# ----------------------------------------------------------------------------------------------------------#
train = label.iloc[:num]
test = label.iloc[num:]


def tpr_weight_funtion(y_true,y_predict):
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
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': 1,
    'is_unbalance': True,
    'lambda_l1': 0.1, #0
    'min_child_weight':4,
    'min_child_samples':5
}

N = 5



test_id = test.UID
test.drop(['UID', 'Tag'], axis=1, inplace=True)


submit = []

# 验证
if 1:
    feature = train.columns.tolist()
    feature.remove('Tag')
    feature.remove('UID')
    X_train = train[~train['UID'].isin(val_UID)]
    y_train = X_train['Tag']
    del X_train['Tag']
    del X_train['UID']
    X_val = train[train['UID'].isin(val_UID)]
    y_val = X_val['Tag']
    del X_val['Tag']
    del X_val['UID']

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=30,
                    verbose_eval=500
                    )
    score = tpr_weight_funtion(y_val, gbm.predict(X_val, num_iteration=gbm.best_iteration))
    print("eval:%f" % score)
    fi = pd.DataFrame(gbm.feature_importance(), index=feature, columns=['fi'])
# 预测
else:
    label = train.Tag
    for k in range(N):
        print('train _K_ flod', k)
        X_train, X_val, y_train, y_val = train_test_split(train, label, test_size=0.2)
        del X_train['Tag']
        del X_train['UID']
        del X_val['Tag']
        del X_val['UID']

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=2000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=30,
                        verbose_eval=500
                        )
        score = tpr_weight_funtion(y_val, gbm.predict(X_val, num_iteration=gbm.best_iteration))
        print("eval:%f" % score)
        submit.append(gbm.predict(test, num_iteration=gbm.best_iteration))

    Tag = 0
    for each in submit:
        Tag += each
    Tag /= N

    res = pd.DataFrame(test_id)
    res['Tag'] = Tag
    res.to_csv('../submit/result.csv', index=False)