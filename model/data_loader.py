import os
import pickle

import numpy as np
import pandas as pd
import keras

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold, StratifiedKFold

import warnings
warnings.filterwarnings("ignore")


class PatchLoader:
    def __init__(self, n_kfold, seed, use_norm=True, server='kakao'):
        self.n_kfold = n_kfold
        self.seed = seed
        
        if server == 'local':
            if use_norm:
                self.patches_mask_path = './data/volume/dataset/level4/img_norm/'
                self.patches_mask_path = './data/volume/dataset/level4/mask_norm/'
                self.img_mask_pairs_path = './data/volume/dataset/level4/img_mask_norm_pairs.pkl'
            else:
                self.patches_mask_path = './data/volume/dataset/level4/img/'
                self.patches_mask_path = './data/volume/dataset/level4/mask/'
                self.img_mask_pairs_path = './data/volume/dataset/level4/img_mask_pairs.pkl'
        elif server == 'kakao':
            if use_norm:
                self.patches_img_path = '/data/volume/dataset/level4/img_norm/'
                self.patches_mask_path = '/data/volume/dataset/level4/mask_norm/'
                self.img_mask_pairs_path = '/data/volume/dataset/level4/img_mask_norm_pairs.pkl'
            else:
                self.patches_img_path = '/data/volume/dataset/level4/img/'
                self.patches_mask_path = '/data/volume/dataset/level4/mask/'
                self.img_mask_pairs_path = '/data/volume/dataset/level4/img_mask_pairs.pkl'
            
    
    def get_all_patches(self):
        '''slide & mask의 pair를 불러와 dataframe으로 만드는 함수'''

        with open(self.img_mask_pairs_path, 'rb') as f:
            img_mask_pairs = pickle.load(f)

        self.all_patches_sample = pd.DataFrame(img_mask_pairs, columns=['slide_path', 'mask_path'])
        self.all_patches_sample = self.all_patches_sample.sample(frac=1, random_state=42).reset_index(drop=True)
        return self.all_patches_sample
    
    
    def split_sample(self):
        kf = KFold(n_splits=self.n_kfold, shuffle=True, random_state=self.seed)
        folds = list(kf.split(self.all_patches_sample))
        return folds

# K-Fold Data Generator
def kfold_data_generator(slide_datagen, mask_datagen, df, batch_size=32, seed=42):
    slide_generator = \
        slide_datagen.flow_from_dataframe(df,
                                          x_col='slide_path',
                                          y_col='mask_path',
                                          seed=seed,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          class_mode=None)

    mask_generator = \
        mask_datagen.flow_from_dataframe(df,
                                         x_col='mask_path',
                                         y_col='mask_path',
                                         color_mode='grayscale',
                                         seed=seed,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         class_mode=None)
    
    
    generator = zip(slide_generator, mask_generator)
    for (slide, mask) in generator:
        mask = mask.astype(np.int8)
        yield slide, mask