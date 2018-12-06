import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
import pandas as pd
import numpy as np
model = Sequential()
model.add(Embedding(1000, 64, embeddings_regularizer='l1', input_length=10))
model.add(Flatten())
# 模型将输入一个大小为 (batch, input_length) 的整数矩阵。
# 输入中最大的整数（即词索引）不应该大于 999 （词汇表大小）
# 现在 model.output_shape == (None, 10, 64)，其中 None 是 batch 的维度。

input_array = np.random.randint(1000, size=(32, 10))

model.compile('adam', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 640)