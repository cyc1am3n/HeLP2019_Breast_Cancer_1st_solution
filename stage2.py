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

def stage2_predict(pd_feature, b_me, b_ma):

    pd_feature_1 = pd_feature.iloc[:,test2_pick_indexes]
    print('test2 meta : ',pd_feature_1.columns)
    x = np.array(pd_feature_1.values)
    scaler2 = StandardScaler()
    x2 = scaler2.fit_transform(x)
    y_me = b_me.predict(x2)

    print('test2 major : ',pd_feature.columns)
    x = np.array(pd_feature.values)
    scaler2 = StandardScaler()
    x2 = scaler2.fit_transform(x)
    y_ma = b_ma.predict(x2)

    y_ma = np.exp(y_ma) - 1
    if len(pd_feature) > 108: # for SNU dataset
      y_ma = y_ma / 1.76
    for i in range(len(y_ma)):
      if y_ma[i] < 500:
        y_ma[i] = 0
  
    return y_me, y_ma
  
  
RESIZE_LIST = [4,16,64]

train_pick_indexes = [1,7,8,9,10, 12,18,19,20,21, 23,29,30,31,32]
test2_pick_indexes = [1,3,4,5,6,  8,9,11,12,13,   15,17,18,19,20]
def stage2_train_meta(pd_feature):
  print('train meta : ',pd_feature.columns)
  
  #pd_feature.drop(pd_feature.columns[drop_indexes], axis='columns', inplace=True)
  pd_feature = pd_feature.iloc[:,train_pick_indexes]
  x = np.array(pd_feature.values)
  scaler2 = StandardScaler()
  x2 = scaler2.fit_transform(x)

  label_df = pd.read_csv(LABEL_PATH)
  y_meta = np.array(label_df.metastasis.tolist())
  y_major_axis = np.array(label_df.major_axis.tolist())
  y_major_axis_log = np.log(y_major_axis + 1) # 나중에 꼭 e 과 1 빼주기

  names = ["RFR"]
  for i in range(len(ml_models)):
    model = ml_models[i]
    model.fit(x2,y_meta)
    auc_score = roc_auc_score(y_meta, model.predict(x2))
    print(names[i],' roc_auc_score : ',auc_score)
  
  print(model.feature_importances_)
  return model


train_pick_indexes_major = [0,1,6,7,8,9,10, 11,12,17,18,19,20,21, 22,23,28,29,30,31,32]
def stage2_train_major(pd_feature):
  
  pd_feature = pd_feature.iloc[:,train_pick_indexes_major]
  print('train major : ',pd_feature.columns)
  #pd_feature.drop(pd_feature.columns[drop_indexes], axis='columns', inplace=True)
  x = np.array(pd_feature.values)
  scaler2 = StandardScaler()
  x2 = scaler2.fit_transform(x)

  label_df = pd.read_csv(LABEL_PATH)
  y_meta = np.array(label_df.metastasis.tolist())
  y_major_axis = np.array(label_df.major_axis.tolist())
  y_major_axis_log = np.log(y_major_axis + 1) # 나중에 꼭 e 과 1 빼주기

  names = ["RFR"]
  for i in range(len(ml_models)):
    model = ml_models[i]
    model.fit(x2,y_major_axis_log)
    pred = model.predict(x2)
    pred = np.exp(pred) - 1

    for i in range(len(pred)):
      if pred[i] < 500:
        pred[i] = 0
    acc_sc = acc_score(y_major_axis, pred)
    print('major_axis, acc score : ',acc_sc)
  
  print(model.feature_importances_)
  return model
  
def check_train_score(pd_feature):
  LABEL_PATH = '/data/train/label.csv'
  label_df = pd.read_csv(LABEL_PATH)
  y_meta = np.array(label_df.metastasis.tolist())
  y_major_axis = np.array(label_df.major_axis.tolist())

  # check train meta score
  cols_name = list(pd_feature.columns)
  best_auc = 0
  best_auc_col = 0
  for i in range(len(cols_name)):

      col_idx = i
      col_name = cols_name[col_idx]
      predict_meta = pd_feature.iloc[:,col_idx].tolist()

      auc_score = roc_auc_score(y_meta, predict_meta)
      print(col_name, 'AUC score : ',auc_score)

      if best_auc < auc_score :
          best_auc = auc_score
          best_auc_col = col_idx
  
  # check train major_axis score
  best_threshold = 0
  best_acc_sc = 0
  major_cols = [0,2,4,6,11,13,15,17, 22,24,26,28]
  for i in range(len(major_cols)):
      col_idx = major_cols[i]
      col_name = cols_name[col_idx] 
      predict_major = np.array(pd_feature.iloc[:, col_idx].tolist()) * 1.757
      
      acc_sc = acc_score(y_major_axis, predict_major)
      print('--------------------------------------')
      print(col_name, 'ACC score : ',acc_sc)
      ## 
      print('----------- set thresholds ----------')

      
      thresholds = [50,100,250, 300,350,400,450,500,550,600,1000]
      for j in range(len(thresholds)):
          threshold = thresholds[j]
          tmp_major = []
          for k in range(len(predict_major)):
              if predict_major[k] < threshold:
                  tmp_major.append(0)
              else :
                  tmp_major.append(predict_major[k])
          acc_sc = acc_score(y_major_axis, tmp_major)
          print(threshold, ' threshold acc_score : ',acc_sc)
          if acc_sc > best_acc_sc :
              best_acc_sc = acc_sc
              best_threshold = threshold
              best_acc_col = col_idx
  return best_auc_col , best_acc_col, best_threshold
