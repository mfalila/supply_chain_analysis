#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 09:59:58 2020

@author: Sam Mfalila
"""
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
import sklearn.preprocessing
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from build_library.utils import FeatureSelector, DropMissing, CategoricalFeatsAdded,\
RemoveNegativeValues, CategoricalImputerTransformer, SimpleImputerTransformer, \
CapOutliers, DelUnusedCols, StandardScalerTransformer


app = Flask(__name__)

clf_checkpoint = joblib.load('backorder_clf_checkpoint.joblib')
clf_pipeline = clf_checkpoint['preprocessing']
clf_model = clf_checkpoint['model']



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''For rendering results on HTML GUI
    '''
    inventory = int(request.form['national_inv'])
    quantity = int(request.form['in_transit_qty'])
    lead = int(request.form['lead_time'])
    forecast = int(request.form['forecast_3_month'])

    features_dict = {'national_inv':[inventory],'in_transit_qty':[quantity],\
                 'lead_time':[lead],'forecast_3_month':[forecast]}  
        
    features_df = pd.DataFrame(data=features_dict)
    final_features = clf_pipeline.fit_transform(features_df)
    prediction = clf_model.predict(final_features)

    if prediction == 1:
        output = 'backorder'
    else:
        output = 'not backorder'

    return render_template('index.html', prediction_text = \
                           'This item will {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
        