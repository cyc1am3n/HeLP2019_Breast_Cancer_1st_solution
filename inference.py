import os 
import time
import random

import numpy as np
import pandas as pd

from utils import *
from stage2 import *

'''CONFIG'''
OUTPUT_PATH = '/data/output/output.csv'
SEED = np.random.randint(1000)
THRESHOLD = 0.5

'''stage 2 '''
MODEL_NAME = 'fpn_model/'

FEATURE_DIR = '/data/volume/feature/'
RESULTS_PATH = '/data/volume/results/'

E1_meta = 'fpn_cjh_major_taeu345.csv'
E2_major = 'fpn_cjh_major_taeu345.csv'

fpn_test1 = 'fpn_cjh_test1_pickcol_meta_taeu8.csv'
fpn_test2 = 'fpn_cjh_test2_pickcol_meta_taeu8.csv'
unet_test1 = 'unet_cjh_test1_pickcol_meta_taeu8.csv'
unet_test2 = 'unet_cjh_test2_pickcol_meta_taeu8.csv'


IS_PREPROCESSED = True # 한번 돌리고 둘다 True 로 바꾸자
IS_FEATURE = True # 한번 돌리고 둘다 True 로 바꾸자
''''''
# print(MODEL_NAME,WEIGHT_NAME)

random.seed(SEED)
np.random.seed(SEED)

# check the meta stasis

# feature full path
#train_feature_path = '/data/volume/feature/fpn_cjh_rescale4_train_feature2_more.csv' 
#test2_feature_path = '/data/volume/feature/fpn_cjh_rescale4_test2_feature2_more.csv'

feature_train_path = FEATURE_DIR + 'fpn_cjh_train_feature1.csv'
feature_test1_path = FEATURE_DIR + 'fpn_cjh_test1_feature1.csv'
feature_test2_path = FEATURE_DIR + 'fpn_cjh_test2_feature1.csv'

train_feature_path = feature_train_path
test2_feature_path = '/data/volume/feature/fpn_cjh_test2_feature_snu.csv'


def main():
    print('Start Inference!')
    print('!!!Stage 2 ENSEMBLE with load meta predict major!!!')

    ## load features
    pd_feature = pd.read_csv(train_feature_path,index_col = [0])
    ## this is for prediction only by A feature.
    best_auc_col , best_acc_col, best_acc_threshold = check_train_score(pd_feature)
    ## the best model of metastasis prediction 2nd stage model 
    best_model_meta = stage2_train_meta(pd_feature)
    ## the best model of major-axis prediction 2nd stage model 
    best_model_major = stage2_train_meta(pd_feature)
    ## load meta
    e1_meta = pd.read_csv(RESULTS_PATH + E1_meta,index_col = [0])
    e2_major = pd.read_csv(RESULTS_PATH + E2_major, index_col = [0])
    e1_meta_list = e1_meta.metastasis.tolist()
    e2_major_list = e2_major.major_axis.tolist()


    ## predict major axis ...
    slide_dir = '/data/test/level4/Image/'
    slide_pathes = sorted(os.listdir(slide_dir))
    
    e1_list = []
    e2_list = []
    cols = [1,3,5,8, 12,14,16,19, 23,25,27,30, 34,36,38,41] 
    current_test_col = cols[3] # 0 ~ 16


    if len(slide_pathes) == 107:
        # test 1 phase
        phase = 'test1'
        print('test1 phase predict...')
    
        #ensemble
        pd_feature = pd.read_csv(FEATURE_DIR + MODEL_NAME[:-1] + '_test1_feature1.csv',index_col = [0])
        fpn_pd = pd.read_csv(RESULTS_PATH + fpn_test1,index_col = [0])
        unet_pd = pd.read_csv(RESULTS_PATH + unet_test1, index_col = [0])

        fpn_meta = fpn_pd.metastasis.tolist()
        unet_meta = unet_pd.metastasis.tolist()

        for i in range(len(fpn_meta)):
            e1_meta_list[i] = 0.5 *(fpn_meta[i] + unet_meta[i])

        e2_major_list = pd_feature.iloc[:,11].tolist()
        
        for i in range(len(e2_major_list)):

            if e2_major_list[i] < 500 :
                e2_major_list[i] = 0
            
    else :
        # test 2 phase - only SNU dataset
        phase = 'test2'
        print('test2 phase predict...')
    
        ## load feature for test 2phase
        pd_feature = pd.read_csv(test2_feature_path,index_col = [0])
            
        ## prediction by only one feature
        best_auc_col = 5 # 4, 5, 11, 12, 18, 19  # max probability value of the given probability heatmap
        best_acc_col = 16 # 0, 2, 7, 9, 14, 16 (best 로 바꿔서 제출)  # major axis of the given probability heatmap
        best_acc_threshold = 500
        e1_meta_list = pd_feature.iloc[:,best_auc_col].tolist() # 
        e2_major_list = np.array(pd_feature.iloc[:,best_acc_col].tolist()) / 1.76

        ## prediction by 2nd stage model
        e1_meta_list, e2_major_list = stage2_predict(pd_feature, best_model_meta, best_model_major)
        
        for i in range(len(e2_major_list)):
            if e2_major_list[i] < best_acc_threshold :
                e2_major_list[i] = 0
        """
        # ensemble
        fpn_pd = pd.read_csv(RESULTS_PATH + fpn_test2,index_col = [0])
        unet_pd = pd.read_csv(RESULTS_PATH + unet_test2, index_col = [0])
        fpn_meta = fpn_pd.metastasis.tolist()
        unet_meta = unet_pd.metastasis.tolist()
        for i in range(len(e1_meta_list)):
            e1_meta_list[i] = 0.5 * (fpn_meta[i] + unet_meta[i])
        """


    total_result = []
    for i, slide_path in enumerate(slide_pathes):
        slide_id = slide_path.split('.')[0]
        total_result.append([slide_id, e1_meta_list[i], e2_major_list[i]])
        print(total_result[i])
    
    result = pd.DataFrame(data=total_result, columns=['id', 'metastasis', 'major_axis'])
    result.to_csv(OUTPUT_PATH, index=False)

    print(SEED)
    save_path = '/data/volume/results/1_'+phase+'_final_' + str(best_auc_col)+'_'+str(best_acc_col)+'.csv'
    result.to_csv(save_path, index=False)
    print(save_path)

if __name__ == "__main__":
    main()
