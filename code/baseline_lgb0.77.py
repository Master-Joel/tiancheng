import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import pickle
import datetime



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


op_train = pd.read_csv('../input/operation_train_new.csv')
trans_train = pd.read_csv('../input/transaction_train_new.csv')

# op_test = pd.read_csv('../input/operation_round1_new.csv')
# trans_test = pd.read_csv('../input/transaction_round1_new.csv')

op_test = pd.read_csv('../input/test_operation_round2.csv')
trans_test = pd.read_csv('../input/test_transaction_round2.csv')
y = pd.read_csv('../input/tag_train_new.csv')
sub = pd.read_csv('../input/submit_example_2.csv')

def getPer(x):
    if x >= 0 and x <= 6:
        return 1
    elif x >= 7 and x <= 12:
        return 2
    elif x >= 13 and x <= 18:
        return 3
    elif x >= 19 and x <= 23:
        return 4



# ----------------------------------------------------------------------------------------------------------------------
###查看  交叉下sum的表现，不行就换成mean std之类

def get_feature(op, trans, label):
    del op['ip2']
    del op['ip2_sub']
    del trans['code2']

    trans['bal_amt'] = trans['bal'] / trans['trans_amt']

    op['hour'] = op['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
    trans['hour'] = trans['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
    op['hour_period'] = op['hour'].apply(lambda x: getPer(x))
    trans['hour_period'] = trans['hour'].apply(lambda x: getPer(x))

    op['hour_period_as'] = (op['day'] - 1) * 4 + op['hour_period']
    trans['hour_period_as'] = (trans['day'] - 1) * 4 + trans['hour_period']

    op['minute'] = op['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').minute)
    trans['minute'] = trans['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').minute)
    # 用户在连续一段时间内的记录视为一次会话
    # 会话次数
    op['timestamp'] = op['day'] * 1440 + op['hour'] * 60 + op['minute']
    trans['timestamp'] = trans['day'] * 1440 + trans['hour'] * 60 + trans['minute']
    op = op.reset_index()
    trans = trans.reset_index()
    tmp1 = op[['index', 'UID', 'timestamp']]
    tmp1['type'] = 0
    tmp2 = trans[['index', 'UID', 'timestamp']]
    tmp2['type'] = 1
    op_tran = pd.concat([tmp1, tmp2])

    op_tran = op_tran.sort_values('timestamp', kind='mergesort')
    op_tran = op_tran.sort_values('UID', kind='mergesort')

    from collections import defaultdict
    action_count = defaultdict(int)

    def count_action(action_count, UID, timestamp):
        if action_count[UID] == 0:
            action_count[UID] = [timestamp, 1]
            return 1
        else:
            if abs(action_count[UID][0] - timestamp) <= 10:
                action_count[UID] = [timestamp, action_count[UID][1]]
            else:
                action_count[UID] = [timestamp, action_count[UID][1] + 1]
            return action_count[UID][1]

    op_tran['period'] = op_tran.apply(lambda x: count_action(action_count, x['UID'], x['timestamp']), axis=1)

    op = op.merge(op_tran[op_tran['type']==0][['index', 'period']],how='left',on='index')
    trans = trans.merge(op_tran[op_tran['type'] == 1][['index', 'period']], how='left', on='index')
    del action_count
    del op_tran


    # 缺失数
    tmp = op.groupby('UID').day.count()
    for feature in ['version', 'device1', 'device2', 'device_code1', 'device_code2', 'device_code3', 'mac1', 'mac2',
                    'ip1',
                    'wifi', 'geo_code', 'ip1_sub']:
        temp = op[op[feature].isna()].groupby('UID').day.count()
        temp1 = temp.reset_index()
        temp1.columns = ['UID', feature + '_NaNum']
        label = label.merge(temp1, on='UID', how='left')
        temp = temp / tmp
        temp.columns = ['UID', feature + '_NaRate']
        label = label.merge(temp.reset_index(), on='UID', how='left')

    tmp = trans.groupby('UID').day.count()
    for feature in ['code1', 'acc_id1', 'device_code1', 'device_code2', 'device_code3', 'device1', 'device2',
                    'mac1', 'amt_src2',
                    'acc_id2', 'acc_id3', 'geo_code', 'trans_type2', 'market_code', 'market_type']:
        temp = trans[trans[feature].isna()].groupby('UID').day.count()
        temp1 = temp.reset_index()
        temp1.columns = ['UID', feature + '_NaNum']
        label = label.merge(temp1, on='UID', how='left')
        temp = temp / tmp
        temp.columns = ['UID', feature + '_NaRate']
        label = label.merge(temp.reset_index(), on='UID', how='left')
    del temp


    label = label.fillna(0)

    del op['hour']
    del trans['hour']
    del op['minute']
    del trans['minute']
    del op['timestamp']
    del trans['timestamp']
    del op['index']
    del trans['index']
    ##
    del op['time']
    del trans['time']
    for feature in op.columns[:]:
        if feature not in ['day', 'hour_period', 'period','hour_period_as']:
            if feature != 'UID':
                label = label.merge(op.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
                label = label.merge(op.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
                                                              ####################
            for deliver in ['ip1', 'mac1', 'mac2', 'geo_code','os','mode','success']:
                if feature not in deliver:
                    if feature != 'UID':
                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                        # temp1 = temp.groupby('UID')[feature].sum().reset_index()
                        # temp1.columns = ['UID', feature + deliver+'_sum']
                        # label = label.merge(temp1, on='UID', how='left')
                        # del temp1
                        temp2 = temp.groupby('UID')[feature].mean().reset_index()
                        temp2.columns = ['UID', feature + deliver]
                        label = label.merge(temp2, on='UID', how='left')
                        del temp2
                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                        # temp1 = temp.groupby('UID')[feature].sum().reset_index()
                        # temp1.columns = ['UID', feature + deliver+'_sum']
                        # label = label.merge(temp1, on='UID', how='left')
                        # del temp1
                        temp2 = temp.groupby('UID')[feature].mean().reset_index()
                        temp2.columns = ['UID', feature + deliver]
                        label = label.merge(temp2, on='UID', how='left')
                        del temp2
                    else:
                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                   how='left')[['UID_x', 'UID_y']]
                        # temp1 = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        # temp1.columns = ['UID', feature + deliver+'_sum']
                        # label = label.merge(temp1, on='UID', how='left')
                        # del temp1
                        temp2 = temp.groupby('UID_x')['UID_y'].mean().reset_index()
                        temp2.columns = ['UID', feature + deliver]
                        label = label.merge(temp2, on='UID', how='left')
                        del temp2
                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                                   how='left')[['UID_x', 'UID_y']]
                        # temp1 = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        # temp1.columns = ['UID', feature + deliver+'_sum']
                        # label = label.merge(temp1, on='UID', how='left')
                        # del temp1
                        temp2 = temp.groupby('UID_x')['UID_y'].mean().reset_index()
                        temp2.columns = ['UID', feature + deliver]
                        label = label.merge(temp2, on='UID', how='left')
                        del temp2
            ##############################
            if feature in ['mode', 'os', 'success', 'version', 'wifi', 'ip1', 'mac1']:
                for deliver in ['period', 'hour_period', 'hour_period_as']:
                    temp = \
                        op[['UID', deliver]].merge(
                            op.groupby([deliver])[feature].count().reset_index(),
                            on=deliver, how='left')[['UID', feature]]
                    # temp1 = temp.groupby('UID')[feature].sum().reset_index()
                    # temp1.columns = ['UID', feature + deliver+'_sum']
                    # label = label.merge(temp1, on='UID', how='left')
                    # del temp1
                    temp2 = temp.groupby('UID')[feature].mean().reset_index()
                    temp2.columns = ['UID', 'op_'+feature + deliver+'_count_mean']
                    label = label.merge(temp2, on='UID', how='left')
                    del temp2
                    temp = \
                        op[['UID', deliver]].merge(
                            op.groupby([deliver])[feature].nunique().reset_index(),
                            on=deliver, how='left')[['UID', feature]]
                    # temp1 = temp.groupby('UID')[feature].sum().reset_index()
                    # temp1.columns = ['UID', feature + deliver+'_sum']
                    # label = label.merge(temp1, on='UID', how='left')
                    # del temp1
                    temp2 = temp.groupby('UID')[feature].mean().reset_index()
                    temp2.columns = ['UID', feature + deliver]
                    label = label.merge(temp2, on='UID', how='left')
                    del temp2
        else:
            print(feature)                             ######
            if feature not in ['day','hour_period_as', 'period']:
                label = label.merge(trans.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].median().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].skew().reset_index(), on='UID', how='left')
            label = label.merge((op.groupby(['UID'])[feature].max() - op.groupby(['UID'])[feature].min()).reset_index(), on='UID', how='left')


            if feature == 'period':
                for deliver in ['day','hour_period_as','hour_period']:
                    temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
            for deliver in ['ip1', 'mac1', 'mac2', 'mode', 'os', 'success','device2']:
                if feature not in deliver:
                    if feature not in ['day','hour_period_as']:
                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].max().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].mean().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')

                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].min().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].mean().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')

                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].sum().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].mean().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')

                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].mean().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].mean().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')

                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                      how='left')[['UID', feature]]
                    # temp = temp.groupby('UID')[feature].sum().reset_index()
                    # temp.columns = ['UID', feature + deliver+'_sum']
                    # label = label.merge(temp, on='UID', how='left')
                    # del temp
                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].std().reset_index(), on=deliver,
                                                      how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = \
                    op[['UID', deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                               how='left')[['UID', feature]]
                    # temp = temp.groupby('UID')[feature].sum().reset_index()
                    # temp.columns = ['UID', feature + deliver+'_sum']
                    # label = label.merge(temp, on='UID', how='left')
                    # del temp
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].std().reset_index(), on=deliver,
                                                      how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp

    for feature in trans.columns[1:]:
        if feature not in ['trans_amt', 'bal', 'bal_amt','day', 'hour_period', 'period', 'hour_period_as']:
            if feature != 'UID':
                label = label.merge(trans.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')

            #for deliver in ['merchant', 'ip1', 'mac1', 'geo_code']:
            for deliver in ['merchant', 'ip1', 'mac1', 'geo_code', 'channel','device2']:
                if feature not in deliver:
                    if feature != 'UID':
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),
                                                             on=deliver, how='left')[['UID', feature]]
                        # temp1 = temp.groupby('UID')[feature].sum().reset_index()
                        # temp1.columns = ['UID', feature + deliver+'_sum']
                        # label = label.merge(temp1, on='UID', how='left')
                        # del temp1
                        temp2 = temp.groupby('UID')[feature].mean().reset_index()
                        temp2.columns = ['UID', feature + deliver]
                        label = label.merge(temp2, on='UID', how='left')
                        del temp2
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),
                                                             on=deliver, how='left')[['UID', feature]]
                        # temp1 = temp.groupby('UID')[feature].sum().reset_index()
                        # temp1.columns = ['UID', feature + deliver+'_sum']
                        # label = label.merge(temp1, on='UID', how='left')
                        # del temp1
                        temp2 = temp.groupby('UID')[feature].mean().reset_index()
                        temp2.columns = ['UID', feature + deliver]
                        label = label.merge(temp2, on='UID', how='left')
                        del temp2
                    else:
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),
                                                             on=deliver, how='left')[['UID_x', 'UID_y']]
                        # temp1 = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        # temp1.columns = ['UID', feature + deliver+'_sum']
                        # label = label.merge(temp1, on='UID', how='left')
                        # del temp1
                        temp2 = temp.groupby('UID_x')['UID_y'].mean().reset_index()
                        temp2.columns = ['UID', feature + deliver]
                        label = label.merge(temp2, on='UID', how='left')
                        del temp2
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),
                                                             on=deliver, how='left')[['UID_x', 'UID_y']]
                        # temp1 = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        # temp1.columns = ['UID', feature + deliver+'_sum']
                        # label = label.merge(temp1, on='UID', how='left')
                        # del temp1
                        temp2 = temp.groupby('UID_x')['UID_y'].mean().reset_index()
                        temp2.columns = ['UID', feature + deliver]
                        label = label.merge(temp2, on='UID', how='left')
                        del temp2
            ##############################
            if feature in ['channel', 'merchant', 'ip1', 'mac1', 'market_code','market_type','trans_type1',
                           'device2', 'acc_id1', 'acc_id2']:
                for deliver in ['period', 'hour_period', 'hour_period_as']:
                    temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),
                                                         on=deliver, how='left')[['UID', feature]]
                    # temp1 = temp.groupby('UID')[feature].sum().reset_index()
                    # temp1.columns = ['UID', feature + deliver+'_sum']
                    # label = label.merge(temp1, on='UID', how='left')
                    # del temp1
                    temp2 = temp.groupby('UID')[feature].mean().reset_index()
                    temp2.columns = ['UID', 'trans_'+feature + deliver+'_count_mean']
                    label = label.merge(temp2, on='UID', how='left')
                    del temp2
                    temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),
                                                         on=deliver, how='left')[['UID', feature]]
                    # temp1 = temp.groupby('UID')[feature].sum().reset_index()
                    # temp1.columns = ['UID', feature + deliver+'_sum']
                    # label = label.merge(temp1, on='UID', how='left')
                    # del temp1
                    temp2 = temp.groupby('UID')[feature].mean().reset_index()
                    temp2.columns = ['UID', feature + deliver]
                    label = label.merge(temp2, on='UID', how='left')
                    del temp2
            label['period_tran_op_rate'] =label['trans_channelperiod_count_mean'] / label['op_modeperiod_count_mean']

            label['hour_period_as_tran_op_rate'] =label['trans_channelhour_period_as_count_mean'] / label['op_modehour_period_as_count_mean']
        else:
            print(feature)
            if feature not in ['bal']:
                if feature not in ['day','hour_period_as']:
                    label = label.merge(trans.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
                    label = label.merge(trans.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left')
                    label = label.merge(trans.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
                    label = label.merge(trans.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left')
                    label = label.merge(trans.groupby(['UID'])[feature].median().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].skew().reset_index(), on='UID', how='left')
                label = label.merge((trans.groupby(['UID'])[feature].max() - trans.groupby(['UID'])[feature].min()).reset_index(), on='UID', how='left')


            if feature == 'period':
                for deliver in ['day','hour_period_as','hour_period']:
                    temp = \
                        trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    temp = \
                        trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                                                        #######################
            for deliver in ['merchant', 'ip1', 'mac1', 'hour_period', 'period','hour_period_as','market_code','market_type']:
                #########################
                if feature not in deliver or (feature in ['trans_amt', 'bal', 'bal_amt'] and deliver in ['hour_period','period']):
                    if feature not in ['day','hour_period_as']:
                        temp = \
                            trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].max().reset_index(),
                                                          on=deliver,
                                                          how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].mean().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                        temp = \
                            trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].min().reset_index(),
                                                          on=deliver,
                                                          how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].mean().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                        temp = \
                            trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].sum().reset_index(),
                                                          on=deliver,
                                                          how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].mean().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                        temp = \
                            trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].mean().reset_index(),
                                                          on=deliver,
                                                          how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].mean().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                    temp = \
                    trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                  how='left')[['UID', feature]]
                    temp1 = temp.groupby('UID')[feature].mean().reset_index()
                    temp1.columns = ['UID', feature + deliver]
                    label = label.merge(temp1, on='UID', how='left')
                    del temp1
                    temp2 = temp.groupby('UID')[feature].std().reset_index()
                    temp2.columns = ['UID', feature + deliver]
                    label = label.merge(temp2, on='UID', how='left')
                    del temp2
                    # temp = temp.groupby('UID')[feature].sum().reset_index()
                    # temp.columns = ['UID', feature + deliver+'_sum']
                    # label = label.merge(temp, on='UID', how='left')
                    # del temp
                    temp = \
                    trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                                  how='left')[['UID', feature]]
                    temp1 = temp.groupby('UID')[feature].mean().reset_index()
                    temp1.columns = ['UID', feature + deliver]
                    label = label.merge(temp1, on='UID', how='left')
                    del temp1
                    temp2 = temp.groupby('UID')[feature].std().reset_index()
                    temp2.columns = ['UID', feature + deliver]
                    label = label.merge(temp2, on='UID', how='left')
                    del temp2
                    # temp = temp.groupby('UID')[feature].sum().reset_index()
                    # temp.columns = ['UID', feature + deliver+'_sum']
                    # label = label.merge(temp, on='UID', how='left')
                    # del temp
                    temp = \
                    trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].std().reset_index(), on=deliver,
                                                  how='left')[['UID', feature]]
                    temp1 = temp.groupby('UID')[feature].mean().reset_index()
                    temp1.columns = ['UID', feature + deliver]
                    label = label.merge(temp1, on='UID', how='left')
                    del temp1
                    temp2 = temp.groupby('UID')[feature].std().reset_index()
                    temp2.columns = ['UID', feature + deliver]
                    label = label.merge(temp2, on='UID', how='left')
                    del temp2


    # 交叉缺失
    for feature in ['code1','device_code1','device_code2','mac1','acc_id2','market_code']:
        for deliver in ['acc_id1']:
            temp = trans[['UID', deliver]].merge(
                (trans.groupby(deliver).day.count() - trans.groupby(deliver)[feature].count()).reset_index()
                , how='left', on=deliver)[['UID', 0]]
            temp1 = temp.groupby('UID')[0].mean().reset_index()
            temp1.columns = ['UID', feature + deliver]
            label = label.merge(temp1, on='UID', how='left')
            del temp1

    # 交叉缺失
    for feature in [ 'device_code1', 'device_code2', 'mac1', 'geo_code']:
        for deliver in ['acc_id2']:
            temp = trans[['UID', deliver]].merge(
                (trans.groupby(deliver).day.count() - trans.groupby(deliver)[feature].count()).reset_index()
                , how='left', on=deliver)[['UID', 0]]
            temp1 = temp.groupby('UID')[0].mean().reset_index()
            temp1.columns = ['UID', feature + deliver]
            label = label.merge(temp1, on='UID', how='left')
            del temp1


    # 交叉缺失
    for feature in ['acc_id1', 'device_code1', 'device_code2', 'device1','mac1','ip1',
                    'amt_src2','acc_id2','geo_code','trans_type2','market_code']:
        for deliver in ['amt_src1']:
            temp = trans[['UID', deliver]].merge(
                (trans.groupby(deliver).day.count() - trans.groupby(deliver)[feature].count()).reset_index()
                , how='left', on=deliver)[['UID', 0]]
            temp1 = temp.groupby('UID')[0].mean().reset_index()
            temp1.columns = ['UID', feature + deliver]
            label = label.merge(temp1, on='UID', how='left')
            del temp1

    # 交叉缺失
    for feature in ['code1', 'acc_id2', 'acc_id3']:
        for deliver in ['merchant']:
            temp = trans[['UID', deliver]].merge(
                (trans.groupby(deliver).day.count() - trans.groupby(deliver)[feature].count()).reset_index()
                , how='left', on=deliver)[['UID', 0]]
            temp1 = temp.groupby('UID')[0].mean().reset_index()
            temp1.columns = ['UID', feature + deliver]
            label = label.merge(temp1, on='UID', how='left')
            del temp1

    # 交叉缺失
    for feature in ['market_type', 'trans_type2', 'geo_code', 'acc_id1','device_code1','device_code2','device_code3',
                        'device1','device2','amt_src2','acc_id2','ip1']:
        for deliver in ['channel']:
            temp = trans[['UID', deliver]].merge(
                (trans.groupby(deliver).day.count() - trans.groupby(deliver)[feature].count()).reset_index()
                , how='left', on=deliver)[['UID', 0]]
            temp1 = temp.groupby('UID')[0].mean().reset_index()
            temp1.columns = ['UID', feature + deliver]
            label = label.merge(temp1, on='UID', how='left')
            del temp1



    # 交叉缺失
    for feature in ['market_type', 'mac1', 'geo_code', 'device1', 'device_code1','device_code2','device_code3']:
        for deliver in ['ip1']:
            temp = trans[['UID', deliver]].merge(
                (trans.groupby(deliver).day.count() - trans.groupby(deliver)[feature].count()).reset_index()
                , how='left', on=deliver)[['UID', 0]]
            temp1 = temp.groupby('UID')[0].mean().reset_index()
            temp1.columns = ['UID', feature + deliver]
            label = label.merge(temp1, on='UID', how='left')
            del temp1

    # 交叉缺失
    for feature in ['market_type']:
        for deliver in ['mac1','device_code1']:
            temp = trans[['UID', deliver]].merge(
                (trans.groupby(deliver).day.count() - trans.groupby(deliver)[feature].count()).reset_index()
                , how='left', on=deliver)[['UID', 0]]
            temp1 = temp.groupby('UID')[0].mean().reset_index()
            temp1.columns = ['UID', feature + deliver]
            label = label.merge(temp1, on='UID', how='left')
            del temp1

    # 交叉缺失
    for feature in ['acc_id1','amt_src2','trans_type2','market_code']:
        for deliver in ['code1']:
            temp = trans[['UID', deliver]].merge(
                (trans.groupby(deliver).day.count() - trans.groupby(deliver)[feature].count()).reset_index()
                , how='left', on=deliver)[['UID', 0]]
            temp1 = temp.groupby('UID')[0].mean().reset_index()
            temp1.columns = ['UID', feature + deliver]
            label = label.merge(temp1, on='UID', how='left')
            del temp1

    # 交叉缺失
    for feature in ['device2', 'device_code3', 'mac1', 'wifi','geo_code']:
        for deliver in ['version']:
            temp = op[['UID', deliver]].merge(
                (op.groupby(deliver).day.count() - op.groupby(deliver)[feature].count()).reset_index()
                , how='left', on=deliver)[['UID', 0]]
            temp1 = temp.groupby('UID')[0].mean().reset_index()
            temp1.columns = ['UID', feature + deliver]
            label = label.merge(temp1, on='UID', how='left')
            del temp1

    # 交叉缺失
    for feature in ['wifi','device2']:
        for deliver in ['ip1']:
            temp = op[['UID', deliver]].merge(
                (op.groupby(deliver).day.count() - op.groupby(deliver)[feature].count()).reset_index()
                , how='left', on=deliver)[['UID', 0]]
            temp1 = temp.groupby('UID')[0].mean().reset_index()
            temp1.columns = ['UID', feature + deliver]
            label = label.merge(temp1, on='UID', how='left')
            del temp1


    print("Done")


    return label



train = get_feature(op_train, trans_train, y)
test = get_feature(op_test, trans_test, sub)

params = {
    'boosting_type': 'gbdt',
    'metric':'auc',
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

# 预测

train = train.drop(['Tag'], axis=1).drop(['UID'], axis=1).fillna(-1)
label = y['Tag']

test_id = test['UID']
test = test.drop(['Tag'], axis=1).drop(['UID'], axis=1).fillna(-1)
#max_depth not 7  try feature_fraction 0.7 num_leaves smaller
lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=64, lambda_l1=3, lambda_l2=5, max_depth=-1,
                               n_estimators=5000, objective='binary', subsample=0.9, feature_fraction=0.77,
                               bagging_freq=1, learning_rate=0.02,
                               random_state=1000, min_child_weight=4, min_data_in_leaf=5, min_split_gain=0)
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []

oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test_id.shape[0])

for index, (train_index, test_index) in enumerate(skf.split(train, label)):
    lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
                  eval_set=[(train.iloc[train_index], label.iloc[train_index]),
                            (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,
                            1]

    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
    sub_preds += test_pred / 5

m = tpr_weight_funtion(y_predict=oof_preds, y_true=label)
print(m)

sub = pd.read_csv('../input/submit_example_2.csv')
sub['Tag'] = sub_preds
sub.to_csv('../submit/result_%s.csv' % str(m), index=False)

#oof_preds 也可以对应每次的预测存起来，特征变化时作为不同模型融合

#oof_preds_分数.csv 和 predict_分数.csv  然后叠加做为特征


# ----------------------------------------------------------------------------------------------------------------------
path = '../input/'
trans_train = pd.read_csv(path + 'transaction_train_new.csv')
y = pd.read_csv(path + 'tag_train_new.csv')
trans_train = trans_train.merge(y, on='UID', how='left')


def find_wrong(trans_train, y, feature):
    black = (trans_train.groupby([feature])['Tag'].sum() / trans_train.groupby([feature])['Tag'].count()).sort_values(
        ascending=False)
    tag_count = trans_train.groupby([feature])['Tag'].count().reset_index()
    black = black.reset_index()
    black = black.merge(tag_count, on=feature, how='left')
    black = black.sort_values(by=['Tag_x', 'Tag_y'], ascending=False)
    return black


Test_trans = pd.read_csv(path + 'transaction_round1_new.csv')
Test_tag = pd.read_csv('../submit/result_%s.csv' % str(m))  # 测试样本
# rule_code = [  '5776870b5747e14e' ,'8b3f74a1391b5427' ,'0e90f47392008def' ,'6d55ccc689b910ee' ,'2260d61b622795fb' ,'1f72814f76a984fa' ,'c2e87787a76836e0' ,'4bca6018239c6201' ,'922720f3827ccef8' ,'2b2e7046145d9517' ,'09f911b8dc5dfc32' ,'7cc961258f4dce9c' ,'bc0213f01c5023ac' ,'0316dca8cc63cc17' ,'c988e79f00cc2dc0' ,'d0b1218bae116267' ,'72fac912326004ee' ,'00159b7cc2f1dfc8' ,'49ec5883ba0c1b0e' ,'c9c29fc3d44a1d7b' ,'33ce9c3877281764' ,'e7c929127cdefadb' ,'05bc3e22c112c8c9' ,'5cf4f55246093ccf' ,'6704d8d8d5965303' ,'4df1708c5827264d' ,'6e8b399ffe2d1e80' ,'f65104453e0b1d10' ,'1733ddb502eb3923' ,'a086f47f681ad851' ,'1d4372ca8a38cd1f' ,'29db08e2284ea103' ,'4e286438d39a6bd4' ,'54cb3985d0380ca4' ,'6b64437be7590eb0' ,'89eb97474a6cb3c6' ,'95d506c0e49a492c' ,'c17b47056178e2bb' ,'d36b25a74285bebb']

black = find_wrong(trans_train, y, 'merchant')
rule_code_1 = black.sort_values(by=['Tag_x', 'Tag_y'], ascending=False).iloc[:50].merchant.tolist()
test_rule_uid = pd.DataFrame(Test_trans[Test_trans['merchant'].isin(rule_code_1)].UID.unique())
pred_data_rule = Test_tag.merge(test_rule_uid, left_on='UID', right_on=0, how='left')
pred_data_rule['Tag'][(pred_data_rule[0] > 0)] = 1
pred_data_rule[['UID', 'Tag']].to_csv('../submit/%s final.csv' % str(m), index=False)