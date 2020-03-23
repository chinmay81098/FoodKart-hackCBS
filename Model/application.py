# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 02:53:18 2019

@author: Avi
"""

from flask import Flask,request,jsonify
import pandas as pd
import tensorflow as tf

from datetime import date
app = Flask(__name__)
model=tf.keras.models.load_model('model.h5')

@app.route('/predict',methods=['POST'])

def predict():
    data=request.get_json()
    prediction=pd.DataFrame(data)
    today= date.today()
    prediction['today_date']=today

    prediction['exp_date']=prediction['exp_date'].astype('datetime64')
    prediction['mfg_date']=prediction['mfg_date'].astype('datetime64')
    prediction['today_date']=prediction['today_date'].astype('datetime64')
    prediction['max_days']=prediction['exp_date'].sub(prediction['mfg_date'],axis=0)
    prediction['days_left']=prediction['exp_date'].sub(prediction['today_date'],axis=0)
    prediction=prediction.drop(['mfg_date','exp_date','today_date'],axis=1)
    prediction['days_left']=prediction['days_left'].astype('str')
    prediction['max_days']=prediction['max_days'].astype('str')
    cols=prediction.columns.tolist()
    cols=cols[1:]+cols[:1]
    prediction=prediction[cols]
    prediction['days_left'] = prediction['days_left'].apply(lambda x : x[:-24])
    prediction['max_days'] = prediction['max_days'].apply(lambda x : x[:-24])
    prediction['days_left']=prediction['days_left'].astype('int')
    prediction['max_days']=prediction['max_days'].astype('int')
    print(prediction)
    prediction_x = model.predict(prediction).flatten()
    print(prediction_x)
    return jsonify({'prediction': str(prediction_x)})
    
    
    
if __name__ == '__main__':
    app.run(port='8000')
    