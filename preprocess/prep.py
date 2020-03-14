import os
import pickle

import numpy as np
import pandas as pd
import openslide
import staintools

from glob import glob
from datetime import datetime
from PIL import Image
from tqdm import tqdm_notebook as tqdm
from openslide.deepzoom import DeepZoomGenerator
from skimage.filters import threshold_otsu, threshold_yen

import warnings
warnings.filterwarnings("ignore")


class Preprocess:
    def __init__(self, 
                 patch_size=256,
                 is_norm=False,
                 target_norm_path=False,
                 mode='train',
                 server='kakao'): ##### add option for calculating auc score
        
        if mode == 'train':
            phase = 'train'
        else:
            phase = 'test'
        
        self.mode = mode
        self.patch_size = patch_size
        self.server = server
        
        if self.server == 'kakao':
            self.slide_dir = f'/data/{phase}/level4/Image/'
            self.mask_dir = f'/data/{phase}/level4/Mask/'
            self.img_mask_pairs_path = '/data/volume/dataset/level4/'
            self.patches_img_path = '/data/volume/dataset/level4/img/'
            self.patches_mask_path = '/data/volume/dataset/level4/mask/'
            if is_norm:
                self.patches_img_norm_path = '/data/volume/dataset/level4/img_norm/'
                self.patches_mask_norm_path = '/data/volume/dataset/level4/mask_norm/'
        elif self.server == 'local':
            self.slide_dir = f'../data/{phase}/level4/Image/'
            self.mask_dir = f'../data/{phase}/level4/Mask/'
            self.img_mask_pairs_path = '../data/volume/dataset/level4/'
            self.patches_img_path = '../data/volume/dataset/level4/img/'
            self.patches_mask_path = '../data/volume/dataset/level4/mask/'
            if is_norm:
                self.patches_img_norm_path = '../data/volume/dataset/level4/img_norm/'
                self.patches_mask_norm_path = '../data/volume/dataset/level4/mask_norm/'
        
        if is_norm:
            print('*'*20, 'Color Normalization : True', '*'*20)
            self.is_norm = is_norm
            self.normalizer = self.stain_norm_func(target_norm_path)
        
        self.mode = 'inference'
        
    
    def _make_directory(self):
        '''학습 시킬 데이터셋(patches)을 저장하는 함수'''

        if self.server == 'local':
            dir_path = '../data/volume'
        elif self.server == 'kakao':
            dir_path = '/data/volume'

        if not os.path.exists(f'{dir_path}/dataset'):
            os.mkdir(f'{dir_path}/dataset')
        if not os.path.exists(f'{dir_path}/dataset/level4'):
            os.mkdir(f'{dir_path}/dataset/level4')
        
        if self.is_norm:
            if not os.path.exists(f'{dir_path}/dataset/level4/img'):
                os.mkdir(f'{dir_path}/dataset/level4/img')        
            if not os.path.exists(f'{dir_path}/dataset/level4/mask'):
                os.mkdir(f'{dir_path}/dataset/level4/mask')
            if not os.path.exists(f'{dir_path}/dataset/level4/img_norm'):
                os.mkdir(f'{dir_path}/dataset/level4/img_norm')        
            if not os.path.exists(f'{dir_path}/dataset/level4/mask_norm'):
                os.mkdir(f'{dir_path}/dataset/level4/mask_norm')
        else:
            if not os.path.exists(f'{dir_path}/dataset/level4/img'):
                os.mkdir(f'{dir_path}/dataset/level4/img')        
            if not os.path.exists(f'{dir_path}/dataset/level4/mask'):
                os.mkdir(f'{dir_path}/dataset/level4/mask')

        print('Created Directories')
        return None
    
    
    def find_patches_from_slide(self, 
                                slide_path, 
                                mask_path, 
                                patch_size=256, 
                                filter_nontissue=True):
        '''
        Returns a DataFrame of all patches in slide
        Args:
            - slide_path: path of slide
            - truth_path: path of truth(mask)
            - patch_size: patch size for samples
            - filter_non_tissue: remove samples no tissue detected
        Returns:
            - samples: patches samples from slide
            - positive: > 0 if tumor else not tumor 0
        '''
        
        with openslide.open_slide(slide_path) as slide:
            tiles = DeepZoomGenerator(slide, tile_size=patch_size, overlap=0, limit_bounds=False)
            if patch_size == 256 :
                size = tiles.level_tiles[tiles.level_count-1]
                # print(f'tile size : {size}')  # (23, 58)
            else :
                size = tiles.level_tiles[tiles.level_count-1 -3]
            thumb_slide = slide.get_thumbnail(size)
            # print(f'thumb_slide size : {thumb_slide.size}')
           

        
        if self.mode == 'train':
            with openslide.open_slide(mask_path) as mask:
                thumb_mask = mask.get_thumbnail(size)  # (23, 58)
                # print(f'thumb_mask size : {thumb_mask.size}')
            
        # ############## is tissue 부분 ##############
        slide4_grey = np.array(thumb_slide.convert('L'))
        binary = slide4_grey < 255  # white = 255 
        slide4_not_white = slide4_grey[binary]  # white = 255
        thresh = threshold_yen(slide4_not_white)
        # thresh = threshold_otsu(slide4_not_white)
        # print(f'current thersh : {thresh}')
        
        height, width = slide4_grey.shape  # (height, width)
        for h in range(height):
            for w in range(width):
                if slide4_grey[h, w] > thresh:
                    binary[h, w] = False
                    
        # create pathces DataFrame
        patches = pd.DataFrame(pd.DataFrame(binary).stack())
        patches['is_tissue'] = patches[0]
        patches = pd.DataFrame(pd.DataFrame(binary).stack(), columns=['is_tissue'])
        patches.loc[:, 'slide_path'] = slide_path
        
        
        # ############## is_tumor 부분 ##############
        if self.mode == 'train':
            truth_img_grey = np.array(thumb_mask.convert('L'))
            positive = truth_img_grey.mean()

            if positive > 0:  # tumor인 경우
                # print('positive(tumor)')
                truth_not_black = truth_img_grey[truth_img_grey > 0]
                try:
                    m_thresh = threshold_otsu(truth_not_black)
                except:
                    m_thresh = 190
                patches_y = pd.DataFrame(pd.DataFrame(truth_img_grey).stack(), columns=['is_tumor'])
                patches_y['is_tumor'] = patches_y['is_tumor'] > m_thresh  # 190  # threshold method를 사용 안한 이유?
                samples = pd.concat([patches, patches_y], axis=1)  # slide의 patches와 mask의 patches_y concat 해주기
            else:
                # print('negative(not tumor)')
                samples = patches
                samples.loc[:, 'is_tumor'] = False

        if self.mode == 'inference':
            # Inference pahse
            positive = 0
            samples = patches
            
        if filter_nontissue == True:  # tissue인 것만 가져오기
            samples = samples[samples['is_tissue']==True]

        samples['tile_loc'] = samples.index.tolist()
        samples.reset_index(inplace=True, drop=True)
        # print(f"samples['is_tumor'].value_counts()\n{samples['is_tumor'].value_counts()}")
        
        return samples, positive

    
    def save_patches(self):
        ''' patches들을 저장하는 함수 '''
        
        prepro_start_time = datetime.now()
        print('='*20, 'Step 1 - create patches', '='*20)
        
        
        # create directory if not exist
        self._make_directory()
        
        # create slide_path, mask_path pair
        slide_path_list = glob(f'{self.slide_dir}*.png')
        mask_path_list = glob(f'{self.mask_dir}*.png')

        slide_path_dict, mask_path_dict = {}, {}
        for slide_path, mask_path in zip(slide_path_list, mask_path_list):
            # slide
            slide_name, _ = os.path.splitext(slide_path)
            slide_idx = slide_name.split('_')[-1]   
            
            # mask
            mask_name, _ = os.path.splitext(mask_path)
            mask_idx = mask_name.split('_')[-1]
            
            # update each dictionary
            slide_path_dict[slide_idx] = slide_path
            mask_path_dict[mask_idx] = mask_path
            
        slide_mask_path_pairs = [(idx, slide_path, mask_path_dict[idx]) 
                                    for idx, slide_path in slide_path_dict.items()]

        slide_mask_pairs = []
        slide_mask_norm_pairs = []
        for cnt, (s_idx, slide_path, mask_path) in enumerate(slide_mask_path_pairs):
            print(f'{slide_path} 패치 추출 중 ...')
            samples, positive = self.find_patches_from_slide(slide_path, mask_path)
            
            if positive:  # tumor인 경우
                samples_pos = samples[samples['is_tumor'] == True]
                samples_neg = samples[samples['is_tumor'] == False]
                total_pos = len(samples_pos)
                sample_num = 100
                if total_pos < 50:
                    sample_num = total_pos * 2
                elif total_pos < 10:
                    sample_num = total_os * 5
                    
                samples_pos = samples_pos.sample(sample_num, random_state=42, replace=True)
                samples_neg = samples_neg.sample(sample_num, random_state=42, replace=True)
                samples = samples_neg.append(samples_pos)
            else:  # non tumor인 경우
                sample_num = 100
                total_neg = len(samples)
                if total_neg > 100:
                    samples = samples.sample(sample_num, random_state=42, replace=True)
                    
            
            with openslide.open_slide(slide_path) as slide:
                with openslide.open_slide(mask_path) as mask:
                    slide_tiles = DeepZoomGenerator(slide, tile_size=self.patch_size, overlap=0, limit_bounds=False)
                    mask_tiles = DeepZoomGenerator(mask, tile_size=self.patch_size, overlap=0, limit_bounds=False)
                    for p_idx, (tile_loc, is_tumor) in enumerate(
                            zip(samples['tile_loc'].tolist(), samples['is_tumor'].tolist())):
                        
                        y, x = tile_loc
                        img = slide_tiles.get_tile(slide_tiles.level_count-1, (x, y))
                        mask = mask_tiles.get_tile(mask_tiles.level_count-1, (x, y))
                        slide_mask_pairs.append((f'{self.patches_img_path}slide{s_idx}_{p_idx}.png', 
                                                    f'{self.patches_mask_path}slide{s_idx}_{p_idx}.png'))
                        img.save(f'{self.patches_img_path}slide{s_idx}_{p_idx}.png')  # slide#_patch#.png
                        mask.save(f'{self.patches_mask_path}slide{s_idx}_{p_idx}.png')
                        if self.is_norm:
                            try:
                                img = np.array(img, dtype=np.uint8)
                                to_transform = staintools.LuminosityStandardizer.standardize(img)
                                img_normed = self.normalizer.transform(to_transform)
                                img_normed = Image.fromarray(img_normed)
                                slide_mask_norm_pairs.append((f'{self.patches_img_norm_path}slide_norm{s_idx}_{p_idx}.png', 
                                                             f'{self.patches_mask_norm_path}slide_norm{s_idx}_{p_idx}.png'))
                                img_normed.save(f'{self.patches_img_norm_path}slide_norm{s_idx}_{p_idx}.png')  # slide#_patch#.png
                                mask.save(f'{self.patches_mask_norm_path}slide_norm{s_idx}_{p_idx}.png')
                            except:
                                continue
            
            if cnt % 5 == 0:
                # img - mask pair 저장하기
                with open(f'{self.img_mask_pairs_path}img_mask_pairs_{cnt}.pkl', 'wb') as f:
                    pickle.dump(slide_mask_pairs, f)
                with open(f'{self.img_mask_pairs_path}img_mask_norm_pairs_{cnt}.pkl', 'wb') as f:
                    pickle.dump(slide_mask_norm_pairs, f)

                            
        
        # img - mask pair 저장하기
        with open(f'{self.img_mask_pairs_path}img_mask_pairs.pkl', 'wb') as f:
            pickle.dump(slide_mask_pairs, f)
        with open(f'{self.img_mask_pairs_path}img_mask_norm_pairs.pkl', 'wb') as f:
            pickle.dump(slide_mask_norm_pairs, f)
            
        prepro_end_time = datetime.now()
        print('preprocessing patches img time : %.1f minutes'%((prepro_end_time - prepro_start_time).seconds/60))
        print('='*50)
        return None
    
    
    def stain_norm_func(self, target_image_path):
        target = staintools.read_image(target_image_path)
        target = staintools.LuminosityStandardizer.standardize(target)
        normalizer = staintools.StainNormalizer(method='vahadane')
        normalizer.fit(target)
        return normalizer
    
    
if __name__ == "__main__":
    preprocess = Preprocess(patch_size=256, 
                            is_norm=True, 
                            target_norm_path='./target_norm.png', 
                            mode='train', 
                            server='local')
    
    preprocess.save_patches()
    
    