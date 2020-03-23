def norm(x):
    return(x-train_stats['mean']/train_stats['std'])
def build_model():
    model=tf.keras.Sequential([
            tf.keras.layers.Dense(64,activation='relu',input_shape=[len(train_x.keys())]),

            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(1)])
    optimizer=tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae','mse'])
    return model
import tensorflow as tf
import pandas as pd

import os
import matplotlib.pyplot as plt
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df=pd.read_csv('dataset.csv')
X=df.drop(['new_price'],axis=1)
Y=df['new_price']
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2,random_state=2)
train_stats=train_x.describe()
train_stats=train_stats.transpose()
print(train_stats)
normed_train=norm(train_x)
normed_test=norm(test_x)
model=build_model()
print(model.summary())
history=model.fit(
        normed_train,train_y,
        epochs=800, validation_split=0.2, verbose=2.0)
loss,mae,mse=model.evaluate(normed_test,test_y,verbose=2.0)
print("loss {:5.2f}".format(mse))

test_predictions=model.predict(normed_test).flatten()

print(test_predictions.astype(int))

print(test_y.head(82))
print(normed_test.head())
predict_x=[
        {'max_days':120,
        'days_left':100,
        'mrp':500}
        ]
a=pd.DataFrame(predict_x)
print(model.predict(a).flatten())

model.save('model.h5')
print('model dumped')

