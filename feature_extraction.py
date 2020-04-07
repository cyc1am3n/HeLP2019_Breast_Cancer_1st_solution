import os
import argparse
import time
import random

import numpy as np
import pandas as pd

from keras import models
import segmentation_models as sm
from segmentation_models import get_preprocessing

from preprocess.prep import Preprocess

from utils import *
from stage2 import *
import staintools
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--is_preprocessed', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='fpn_model')
    parser.add_argument('--model_weight', type=str, default='2_fold_fpn_best_model.h5')
    parser.add_argument('--ckpt_dir', type=str, default='/data/volume/model/')
    parser.add_argument('--heatmap_dir', type=str, default='/data/volume/heatmap/')
    parser.add_argument('--feature_dir', type=str, default='/data/volume/feature/')
    parser.add_argument('--patches_dir', type=str, default='/data/volume/patches/rescale/')
    args = parser.parse_args()

    TRAIN_DIR, LABEL_PATH = '/data/train', '/data/train/label.csv'
    MODEL_NAME, HEATMAP_DIR, FEATURE_DIR, PATCHES_DIR = args.model_name, args.heatmap_dir, args.feature_dir, args.patches_dir

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # check isdir
    set_directory(CKPT_DIR, MODEL_NAME)
    make_directory(HEATMAP_DIR)
    make_directory(FEATURE_DIR)
    make_directory(PATCHES_DIR)

    #load model
    MODEL_PATH = args.ckpt_dir + args.model_name + '/' + args.model_weight
    model = models.load_model(
        MODEL_PATH,
        custom_objects={
            'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss,
            'iou_score': sm.metrics.iou_score,
            'f1-score': sm.metrics.f1_score
        }
    )
    print(MODEL_PATH,'Model loaded.')

    # set preprocess
    preprocess_input = get_preprocessing('resnet34')
    preprocess = Preprocess(patch_size=PATCH_SIZE, mode='inference', server='kakao')
    
    TARGET_NORM_PATH = './preprocess/target_norm.png'
    normalizer = stain_norm_func(TARGET_NORM_PATH)
    slide_pathes = sorted(os.listdir(preprocess.slide_dir))
    stain_patches_save_path, phase = stain_patch_dir(PATCHES_DIR, slide_pathes)

    start_time = time.time()
    full_feature_list = []
    
    for i, slide_path in enumerate(slide_pathes):
        current_save_dir = stain_patches_save_path + slide_path[:-4] + '/' # ex) '/data/volume/patches/rescale/test1/slide_001/'
        
        if phase == 'test1' and i <= 60: # AMC dataset
            full_slide_path = preprocess.slide_dir + slide_path
        else : # SNU dataset
            full_slide_path = '/data/test/level0/'+ slide_path +'.mrxs'

        print(current_save_dir)
        if IS_PREPROCESSED : 
            stain_patches_names = sorted(os.listdir(current_save_dir))
        else : 
            make_directory(current_save_dir)

        with openslide.open_slide(full_slide_path) as slide:
            if slide.dimensions[1] < 20000:
                print('AMC data!')
                patch_size = 256
            else :
                print('SNU data!')
                patch_size = 290
            

            slide_tiles = DeepZoomGenerator(slide, tile_size = patch_size, overlap = 0 , limit_bounds = False)
            if patch_size == 290:
                output_preds = np.zeros((int((slide.dimensions[1] / 8 + 1)/1.13), int((slide.dimensions[0] / 8 + 1)/1.13)))
            else: ### snu resolution
                output_preds = np.zeros((slide.dimensions[1],slide.dimensions[0]))
            print('output_preds shape : ',output_preds.shape)
            samples, _ = preprocess.find_patches_from_slide(slide_path = full_slide_path, mask_path = None, patch_size = patch_size) 
            print(samples.is_tissue.value_counts())
            cnt = 0
            for idx, batch_sample in samples.iterrows():
                is_tissue = batch_sample.is_tissue
                x,y = batch_sample.tile_loc[::-1]
                if is_tissue : 
                    if patch_size == 290:
                        img = slide_tiles.get_tile(slide_tiles.level_count-1 -3,(x,y)) # SNU -> level 3
                    else :
                        img = slide_tiles.get_tile(slide_tiles.level_count-1,(x,y))
                    if (img.size == (patch_size, patch_size)): 
                        if IS_PREPROCESSED:
                            try :
                                full_stain_patches_path = current_save_dir + str(idx) + '.png'
                                cnt += 1
                                img = Image.open(full_stain_patches_path)
                                X = np.array(img, dtype =np.uint8)
                            except:
                                X = np.zeros((256,256,3))
                        else :
                            if img.size[0] == 290 : 
                                img = img.resize((256,256))
                                X = np.array(img, dtype = np.uint8)
                                try : 
                                    X = staintools.LuminosityStandardizer.standardize(X)
                                    X = normalizer.transform(X)
                                    x_img = Image.fromarray(X)
                                    x_img.save(current_save_dir + str(idx) + '.png')
                                except:
                                    X = np.zeros((256, 256,3))
                            else :
                                try :
                                    full_stain_patches_path = current_save_dir + str(idx) + '.png'
                                    cnt += 1
                                    img = Image.open(full_stain_patches_path)
                                    X = np.array(img, dtype =np.uint8)
                                except :
                                    X = np.zeros((256,256, 3))

                        X = X.astype(np.float32)
                        X = preprocess_input(X)

                        pred_j = predict_from_model(X, model)

                        '''fill output_preds : full heatmap'''
                        new_x, new_y = batch_sample.tile_loc[0] * 256, batch_sample.tile_loc[1] * 256
                        output_preds[new_x:new_x+256, new_y:new_y+256] = pred_j
        '''make different level heatmaps / input : full size heatmap / output : different scale heatmap'''
        heatmaps_list = make_different_level_heatmaps(output_preds)
        '''extract feature from different level heatmaps'''
        feature_list, feature_name_list = extract_feature_from_heatmaps(heatmaps_list)
        if i == 0:
            print(feature_name_list)
        print(feature_list)
        full_feature_list.append(feature_list)

    pd_feature = pd.DataFrame(np.array(full_feature_list), columns=feature_name_list)
    save_feature_path = FEATURE_DIR + MODEL_NAME +'_' +phase+'_feature.csv'
    pd_feature.to_csv(save_feature_path)
