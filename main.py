

import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

df = pd.read_csv("monthly-milk-production-pounds.csv" , index_col = "Month" ,
                 parse_dates = True)
df.index.freq = "MS"

train = df.iloc[:156]
test = df.iloc[156:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 3
n_features = 1
generator = TimeseriesGenerator(scaled_train,scaled_train,length=n_input,batch_size=1)

x,y = generator[0]
print(f"Given the Array: \n{x.flatten()}")
print(f"Predict this y : \n{y}")

n_input =12
generator = TimeseriesGenerator(scaled_train,scaled_train,length=n_input,batch_size=1)

model = Sequential()
model.add(LSTM(100, activation = "relu" , input_shape=(n_input,n_features)))
model.add(Dense(1))
model.compile(optimizer="adam" , loss = "mse")
model.summary()
model.fit(generator , epochs=5)