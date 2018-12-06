import pandas as pd
import datetime
import numpy as np
import lightgbm as lgb
from  matplotlib import pyplot as plt
from sklearn import preprocessing
import pickle
pd.set_option('display.max_rows',100)

pd.set_option('display.width',1000)
operation_trn = pd.read_csv('../input/operation_TRAIN_new.csv')

transaction_trn = pd.read_csv('../input/transaction_TRAIN_new.csv')

tag_trn = pd.read_csv('../input/tag_TRAIN_new.csv')

operation_test = pd.read_csv('../input/operation_round1_new.csv')
transaction_test = pd.read_csv('../input/transaction_round1_new.csv')
tag_test = pd.read_csv('../input/submission_sample.csv')

op = pd.concat([operation_trn, operation_test])
tran = pd.concat([transaction_trn, transaction_test])
label = pd.concat([tag_trn, tag_test])

num = tag_trn.shape[0]
#operation_trn = pd.merge(operation_trn,tag_trn,how='left',on='UID')
#transaction_trn = pd.merge(transaction_trn,tag_trn,how='left',on='UID')

op['hour'] = op['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
# operation_trn['minute'] = operation_trn['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').minute)
tran['hour'] = tran['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
# transaction_trn['minute'] = transaction_trn['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').minute)
op.drop('time',inplace=True,axis=1)
tran.drop('time',inplace=True,axis=1)
a = set(operation_trn.columns)
b = set(transaction_trn.columns)





train = label.iloc[:num]
test = label.iloc[num:]
test_id = test.UID
del train['UID']
label = train['Tag']
del train['Tag']
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
from sklearn.model_selection import  train_test_split



N=5
submit = []
feature = train.columns.tolist()
for k in range(N):
    print('train _K_ flod', k)
    X_train, X_val, y_train, y_val = train_test_split(train, label, test_size=0.2)

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

r = 0
for each in submit:
    r+=each
r/=N
res = pd.DataFrame(test_id)
res['Tag'] = r
res.to_csv('../submit/result.csv', index=False)


asd
operation_trn['os'] = pd.factorize(operation_trn['os'])[0]
operation_trn['version'] = pd.factorize(operation_trn['version'])[0]


operation_trn['mode'], map = pd.factorize(operation_trn['mode'])
# 黑白样本在操作类型上的差异
plt.title('mode')
a = operation_trn[operation_trn.Tag==0]['mode'].value_counts().sort_index()
x = a.index.tolist()[:35]
y = a.values.reshape(-1,1)[:35]
min_max_scaler = preprocessing.MinMaxScaler()

y = min_max_scaler.fit_transform(y)

plt.plot(x,y,'go')

a = operation_trn[operation_trn.Tag==1]['mode'].value_counts().sort_index()
x = a.index.tolist()[:35]
y1 = a.values.reshape(-1,1)[:35]
min_max_scaler = preprocessing.MinMaxScaler()
y1 = min_max_scaler.fit_transform(y1)

plt.plot(x,y1,'ro')

plt.show()

diff_mode = []
for i in range(len(y1)):
    if abs(y1[i]-y[i])/y1[i] > 2:
        diff_mode.append(map[i])



# 黑白样本在成功数上差异
plt.title('success')
a = operation_trn[operation_trn.Tag==0]['success'].value_counts().sort_index()
x = a.index.tolist()
y = a.values
plt.plot(x,y,'g')

a = operation_trn[operation_trn.Tag==1]['success'].value_counts().sort_index()
x = a.index.tolist()
y = a.values
plt.plot(x,y,'r')

plt.show()

# 黑白样本在操作第几天上差异
plt.title('day')
a = operation_trn[operation_trn.Tag==0]['day'].value_counts().sort_index()
x = a.index.tolist()
y = a.values.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
y = min_max_scaler.fit_transform(y)
plt.plot(x,y,'g')

a = operation_trn[operation_trn.Tag==1]['day'].value_counts().sort_index()
x = a.index.tolist()
y = a.values.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
y = min_max_scaler.fit_transform(y)
plt.plot(x,y,'r')

plt.show()

# 黑白样本在软件版本上差异
plt.title('version')
a = operation_trn[operation_trn.Tag==0]['version'].value_counts().sort_index()


y = a.values.reshape(-1,1)
x = a.index.tolist()
min_max_scaler = preprocessing.MinMaxScaler()
y = min_max_scaler.fit_transform(y)
plt.plot(x,y,'g')

a = operation_trn[operation_trn.Tag==1]['version'].value_counts().sort_index()

y = a.values.reshape(-1,1)
x = a.index.tolist()
min_max_scaler = preprocessing.MinMaxScaler()
y = min_max_scaler.fit_transform(y)
plt.plot(x,y,'r')

plt.show()

# 黑白样本在操作小时上差异
plt.title('hour')
a = operation_trn[operation_trn.Tag==0]['hour'].value_counts().sort_index()


y = a.values.reshape(-1,1)
x = a.index.tolist()
min_max_scaler = preprocessing.MinMaxScaler()
y = min_max_scaler.fit_transform(y)
plt.plot(x,y,'g')

a = operation_trn[operation_trn.Tag==1]['hour'].value_counts().sort_index()

y = a.values.reshape(-1,1)
x = a.index.tolist()
min_max_scaler = preprocessing.MinMaxScaler()
y = min_max_scaler.fit_transform(y)
plt.plot(x,y,'r')

plt.show()

# 黑白样本在操作系统上差异
plt.title('os')
a = operation_trn[operation_trn.Tag==0]['os'].value_counts().sort_index()

x = a.index.tolist()
y = a.values.reshape(-1,1)

plt.plot(x,y,'go')

a = operation_trn[operation_trn.Tag==1]['os'].value_counts().sort_index()
x = a.index.tolist()
y = a.values.reshape(-1,1)

plt.plot(x,y,'ro')

plt.show()


for each in ['channel','amt_src1','trans_type1','amt_src2','trans_type2','market_type']:
    transaction_trn[each] = pd.factorize(transaction_trn[each])[0]
    # 黑白样本在交易平台上差异
    plt.title(each)
    a = transaction_trn[transaction_trn.Tag==0][each].value_counts().sort_index()

    x = a.index.tolist()
    y = a.values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    y = min_max_scaler.fit_transform(y)
    plt.plot(x,y,'g')

    a = transaction_trn[transaction_trn.Tag==1][each].value_counts().sort_index()
    x = a.index.tolist()
    y = a.values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    y = min_max_scaler.fit_transform(y)
    plt.plot(x,y,'r')

    plt.show()



#########################################
a = operation_trn[['UID','day','time','mode']]
a['mode'] = pd.factorize(a['mode'])[0]


b = transaction_trn[['UID','day','time','merchant','amt_src1']]

a['type'] = 0
b['type'] = 1

a = pd.concat([a,b])
a = a = pd.merge(a,tag_trn,how='left',on='UID')
a['merchant'] = pd.factorize(a['merchant'])[0]
a['amt_src1'] = pd.factorize(a['amt_src1'])[0]

a = a.sort_values('time',kind='mergesort')
a = a.sort_values('day',kind='mergesort')
a = a.sort_values('UID',kind='mergesort')




# 对于各个特征，寻找其在黑白样本分布差别过大的取值，单独做为特征，计算用户的取这个值的比例

op_fe = list(a)
tran_fe = list(b)
op_tran_fe = list(a&b)

U = []
U_w = []
U_b = []
S = 0
for flag in range(1,4):
    if flag == 1:
        op_tran = operation_trn[op_fe]
        fe = op_fe
    elif flag == 2:
        op_tran = transaction_trn[tran_fe]
        fe = tran_fe
    else:
        op_tran = pd.concat([operation_trn[op_tran_fe], transaction_trn[op_tran_fe]])
        fe = op_tran_fe

    fe.remove('Tag')
    # ip1暂时不处理，太多
    fe.remove('ip1')
    fe.remove('UID')
    w_num = op_tran[op_tran['Tag']==0].shape[0]
    b_num = op_tran[op_tran['Tag']==1].shape[0]
    uni = {}
    uni_w = {}
    uni_b = {}
    for each in fe:
        print(each+' start')
        a = op_tran[op_tran.Tag == 0][each].value_counts().sort_index()
        b = op_tran[op_tran.Tag == 1][each].value_counts().sort_index()


        x1 = a.index.tolist()
        x2 = b.index.tolist()

        u0 = set(x1) - set(x2)
        u1 = set(x2) - set(x1)
        uni_w[each] = u0
        uni_b[each] = u1
        S+=len(u0)
        S+=len(u1)
        for e in u0:
            a.drop(e,inplace=True)
        for e in u1:
            b.drop(e,inplace=True)


        a = a.sort_index()
        b = b.sort_index()
        x1 = a.index.tolist()
        # min_max_scaler = preprocessing.MinMaxScaler()
        # y1 = min_max_scaler.fit_transform(a.values.reshape(-1,1))
        y1 = a.values/w_num
        x2 = b.index.tolist()
        # min_max_scaler = preprocessing.MinMaxScaler()
        # y2 = min_max_scaler.fit_transform(b.values.reshape(-1,1))
        y2 = b.values/b_num
        assert x1 == x2
        diff = []
        for i in range(len(x1)):
            if abs(y1[i] - y2[i]) > 0.01:
                diff.append(x1[i])
        S += len(diff)
        print(S)
        uni[each] = diff
        print(each + ' end')
    U.append(uni)
    U_w.append(uni_w)
    U_b.append(uni_b)


save_file=open("Uni.bin","wb")
pickle.dump(U,save_file)
pickle.dump(U_w,save_file)
pickle.dump(U_b,save_file)
save_file.close()