import pandas as pd
import numpy as np

operation_trn = pd.read_csv('../input/operation_TRAIN_new.csv')
operation_test = pd.read_csv('../input/operation_round1_new.csv')
transaction_trn = pd.read_csv('../input/transaction_TRAIN_new.csv')
transaction_test = pd.read_csv('../input/transaction_round1_new.csv')
tag_trn = pd.read_csv('../input/tag_TRAIN_new.csv')
tag_test = pd.read_csv('../input/submission_sample.csv')

operation_trn.sort_values('day',kind='mergesort').sort_values('UID',kind='mergesort')