import os
import argparse
import time
import random

import numpy as np
import pandas as pd
import keras

from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from segmentation_models import get_preprocessing

from utils import set_directory
from preprocess.prep import Preprocess
from model.net import fpn, unet
from model.data_loader import PatchLoader
from model.data_loader import kfold_data_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--model', type=str, default='fpn')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--n_folds', type=int, default=3)
    parser.add_argument('--preprocess', type=bool, default=True)
    parser.add_arguemtn('--stain_norm', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt_dir', type=str, default='/data/volume/model/')
    parser.add_argument('--model_name', type=str, default='fpn_model/')
    args = parser.parse_args()

    TRAIN_DIR, LABEL_PATH = '/data/train', '/data/train/label.csv'
    CKPT_DIR, MODEL_NAME = args.ckpt_dir, args.model_name

    random.seed(args.seed)
    np.random.seed(args.seed)

    # check isdir
    set_directory(CKPT_DIR, MODEL_NAME)

    # preprocessing
    if PREPROCESS:
        preprocess = Preprocess(patch_size=args.patch_size, 
                                is_norm=args.stain_norm,
                                target_norm_path='./preprocess/target_norm.png',
                                mode='train',
                                server='kakao')
        preprocess.save_patches()
    else:
        print('Already Preprocessed.')

    # set dataset
    patch_loader = PatchLoader(n_kfold=args.n_folds, 
                               seed=args.seed, 
                               use_norm=args.stain_norm, 
                               server='kakao')
    all_patches_sample = patch_loader.get_all_patches()
    folds = patch_loader.split_sample()

    # set generator
    print('Set Generator.')
    preprocess_input = get_preprocessing('resnet34')

    # Slide, Mask ImageDataGenerator
    train_slide_datagen = ImageDataGenerator(# rescale= 1./255,
                                            width_shift_range=[-10, 10],
                                            rotation_range=90, 
                                            fill_mode='reflect',
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            preprocessing_function=preprocess_input)

    train_mask_datagen = ImageDataGenerator(rescale= 1./255,
                                            width_shift_range=[-10, 10],
                                            rotation_range=90,
                                            fill_mode='reflect',
                                            horizontal_flip=True,
                                            vertical_flip=True)


    valid_slide_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_mask_datagen = ImageDataGenerator(rescale= 1./255)
    
    # set model
    if args.model == 'fpn':
        model = fpn(backbone='resnet34', pretrained_weights='imagenet')
    else:
        model = unet(backbone='resnet34', pretrained_weights='imagenet')

    train_start_time = time.time()
    for f_idx, (train_idx, valid_idx) in enumerate(folds):
        print('*'*20, f'{f_idx}-Fold 학습 시작', '*'*20)
        train_df = all_patches_sample.iloc[train_idx]
        valid_df = all_patches_sample.iloc[valid_idx]
        
        train_slide_mask_gen = kfold_data_generator(train_slide_datagen, 
                                                    train_mask_datagen, 
                                                    df=train_df,
                                                    batch_size=args.batch_size,
                                                    seed=args.seed)

        valid_slide_mask_gen = kfold_data_generator(valid_slide_datagen, 
                                                    valid_mask_datagen, 
                                                    df=valid_df,
                                                    batch_size=args.batch_size,
                                                    seed=args.batch_size)
        
        train_steps = len(train_df) // args.batch_size
        valid_steps = len(valid_df) // args.batch_size
        
        # callbacks_list
        callbacks_list = [
            ModelCheckpoint(
                filepath=f'{CKPT_DIR}{MODEL_NAME}{f_idx+1}_fold_{args.model}_best_model.h5',
                monitor='val_iou_score',
                mode='max',
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor='val_iou_score',
                mode='max',
                factor=0.1,
                patience=3,
                verbose=1,
            )
        ]
        
        history = model.fit_generator(train_slide_mask_gen, 
                                    steps_per_epoch=train_steps,
                                    validation_data=valid_slide_mask_gen,
                                    validation_steps=valid_steps,
                                    epochs=args.epochs, verbose=2,
                                    callbacks=callbacks_list)
        
        model.save(f'{CKPT_DIR}{MODEL_NAME}{args.model}_im_{f_idx+1}_fold_last_model.h5')
        
        print('*'*20, f'{f_idx}-Fold 학습 완료', '*'*20)
        print('='*60)
    
    train_end_time = time.time()
    print('Train time : ', (train_end_time - train_start_time) / 60, 'minutes')
    model.save(f'{CKPT_DIR}{MODEL_NAME}{args.model}_{N_KFOLD}_fold_total_model.h5')
    print('model save completed')