import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge,LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.externals import joblib
from PIL import Image
import random
from skimage.measure import block_reduce
import cv2
from utils import * # get_major_axis, acc_score
from sklearn.metrics import roc_auc_score


RESIZE_LIST = [4,16,64]

def make_different_level_heatmaps(output_preds):
  output_preds = np.array(output_preds)
  heatmaps_list = []
  resize_list = RESIZE_LIST
  for i in resize_list:
    heatmaps_list.append(block_reduce(output_preds, (i,i), np.mean))
  return heatmaps_list

def extract_feature_from_heatmaps(heatmaps_list):
  THRESHOLDS = [0.2,0.5] ##
  resize_list = RESIZE_LIST
  feature_list = []
  feature_name_list = []
  for i, heatmap in enumerate(heatmaps_list):
    for threshold in THRESHOLDS:
      test_np = (heatmap > threshold).astype(np.uint8)
      #kernel = np.ones((resize_list[i], resize_list[i]), np.uint8)
      #test_np = cv2.morphologyEx(test_np, cv2.MORPH_CLOSE, kernel)
      mx_i = get_major_axis(test_np)
      feature_name_list.append(str(resize_list[i]) + '_major_axis_t' + str(threshold))
      feature_list.append(mx_i * resize_list[i])
      tumor_len = np.sum(heatmap > threshold)
      
      tissue_len = np.sum(heatmap > 0.0)
      feature_name_list.append(str(resize_list[i]) + '_tumor_ratio_t' + str(threshold))
      if tissue_len != 0:
        feature_list.append(tumor_len / tissue_len)
      else :
        feature_list.append(0.0)
    
    feature_name_list.append(str(resize_list[i]) + '_max')
    feature_name_list.append(str(resize_list[i]) + '_mean')
    feature_name_list.append(str(resize_list[i]) + '_std')
    feature_list.append(np.max(heatmap))
    feature_list.append(np.mean(heatmap))
    feature_list.append(np.std(heatmap))

  return feature_list, feature_name_list

def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


ml_models = [
          RandomForestRegressor(),
          ]

ROOT_DIR = ''
LABEL_PATH = ROOT_DIR + '/data/train/label.csv' ####### 
OUTPUT_PATH = ROOT_DIR + '/data/output/output.csv'
CKPT_DIR = ROOT_DIR + '/data/volume/model/'
HIST_DIR = ROOT_DIR + '/data/volume/history/'
FEAT_DIR = ROOT_DIR + '/data/volume/feature/'
ML_DIR = CKPT_DIR + 'stage2/'
make_directory(ML_DIR)


drop_columns  = ['4_major_axis_t0.5', '4_tumor_ratio_t0.5', '4_major_axis_t0.9','4_tumor_ratio_t0.9',
                '16_major_axis_t0.5', '16_tumor_ratio_t0.5', '16_major_axis_t0.9', '16_tumor_ratio_t0.9',
                  '64_major_axis_t0.5','64_tumor_ratio_t0.5', '64_major_axis_t0.9', '64_tumor_ratio_t0.9',
                 '256_major_axis_t0.5','256_tumor_ratio_t0.5', '256_major_axis_t0.9', '256_tumor_ratio_t0.9'
                 ]
drop_indexes = [2, 3,  9, 10, 16, 17, 23, 24]

def stage2_train(pd_feature):
  print(pd_feature.columns)
  pd_feature.drop(pd_feature.columns[drop_indexes], axis='columns', inplace=True)
  x = np.array(pd_feature.values)
  scaler2 = StandardScaler()
  x2 = scaler2.fit_transform(x)

  label_df = pd.read_csv(LABEL_PATH)
  y_meta = np.array(label_df.metastasis.tolist())
  y_major_axis = np.array(label_df.major_axis.tolist())
  y_major_axis_log = np.log(y_major_axis + 1) # 나중에 꼭 e 과 1 빼주기

  names = [ "RF"]
  for i in range(len(ml_models)):
    model = ml_models[i]
    model.fit(x2,y_meta)
    auc_score = roc_auc_score(y_meta, model.predict(x2))
    print(names[i],' roc_auc_score : ',auc_score)

    model.fit(x2,y_major_axis_log)
    pred = model.predict(x2)
    pred = np.exp(pred) - 1
    thresholds = [50, 100, 250, 500, 1000]
    pred_tmp = pred.copy()
    for thresh in thresholds:
      for j in range(len(pred)):
        if pred[j] < thresh :
          pred_tmp[j] = 0
      acc_sc = acc_score(y_major_axis, pred_tmp)
      print(names[i], thresh, 'thresh value, acc score : ',acc_sc)

def stage2(pd_feature):
    ## x 
    pd_feature.drop(pd_feature.columns[drop_indexes], axis='columns', inplace=True)
    #pd_feature.drop(drop_columns, axis='columns', inplace=True)
    x = np.array(pd_feature.values)
    scaler1 = MinMaxScaler()
    scaler2 = StandardScaler()
    x1 = scaler1.fit_transform(x)
    x2 = scaler2.fit_transform(x)

    ## y
    label_df = pd.read_csv(LABEL_PATH)
    y_meta = np.array(label_df.metastasis.tolist())
    y_major_axis = np.array(label_df.major_axis.tolist())
    y_major_axis_log = np.log(y_major_axis + 1) # 나중에 꼭 e 과 1 빼주기

    names = [ "RF"]
    models_ml_len = len(ml_models)
    random_model_index = np.random.randint(models_ml_len)
    random_model_meta = ml_models[random_model_index]
    random_model_meta.fit(x2,y_meta)
    auc_score = roc_auc_score(y_meta, random_model_meta.predict(x2))
    print(names[random_model_index],' roc_auc_score : ',auc_score)
    print(random_model_meta)

    random_model_index = np.random.randint(models_ml_len)
    random_model_major = ml_models[random_model_index]
    random_model_major.fit(x2, y_major_axis_log)
    pred = random_model_major.predict(x2)
    pred = np.exp(pred) - 1
    for i in range(len(pred)):
        pred[i] = 0
    acc_sc = acc_score(y_major_axis, pred)
    print(random_model_major)
    print('major_axis all 0, acc score : ',acc_sc)

    return random_model_meta, random_model_major

def stage2_predict(pd_feature, m_me, m_ma):
    #pd_feature.drop(drop_columns, axis='columns', inplace=True)
    pd_feature.drop(pd_feature.columns[drop_indexes], axis='columns', inplace=True)
    print(pd_feature)
    x = np.array(pd_feature.values)
    scaler1 = MinMaxScaler()
    scaler2 = StandardScaler()

    if len(pd_feature.values) == 107:
      x1 = scaler2.fit_transform(x[:61])
      x2 = scaler2.fit_transform(x[61:])
      y_me_1 = m_me.predict(x1)
      y_me_2 = m_me.predict(x2)

      y_me = np.append(y_me_1, y_me_2)
  
    else : 
      x2 = scaler2.fit_transform(x)
      y_me = m_me.predict(x2)
    
    x2 = scaler2.fit_transform(x)
    y_ma = m_ma.predict(x2)

    y_ma = np.exp(y_ma) - 1
    
    for i in range(len(y_ma)):
        y_ma[i] = 0
    
    print(len(y_me), len(y_ma))
    return y_me, y_ma