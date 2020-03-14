import os
import numpy as np
import staintools
from skimage.measure import label, regionprops

def stain_norm_func(target_image_path):
    target = staintools.read_image(target_image_path)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)
    return normalizer

def stain_patch_dir(PATCHES_DIR, slide_pathes):
    phase = 'train'
    stain_patches_save_path = PATCHES_DIR + 'train/'
    if len(slide_pathes) < 110 :
        phase = 'test1'
        stain_patches_save_path = PATCHES_DIR + 'test1/'
    elif len(slide_pathes) < 200 :
        phase = 'test2'
        stain_patches_save_path = PATCHES_DIR + 'test2/'
    make_directory(stain_patches_save_path)
    print('current phase : ',phase)
    return stain_patches_save_path, phase


def set_directory(CKPT_DIR, MODEL_NAME):
    if not os.path.isdir(CKPT_DIR):
        os.mkdir(CKPT_DIR)
    if not os.path.isdir(CKPT_DIR + MODEL_NAME):
        os.mkdir(CKPT_DIR + MODEL_NAME)
    print('Set Directory')

    
def get_major_axis(mask):
    from skimage.measure import label, regionprops
    
    # divide entire masks into each instance using connected-components labelling
    labels = label(mask)
    
    # iterate to calculate the length of the major axis of each instance
    major_axis_list = [regionprops((labels == i).astype('uint8'))[0].major_axis_length \
                       for i in np.unique(labels) if i != 0]
    
    # find the longest major axis
    if len(major_axis_list):
        longest_major_axis = max(major_axis_list)
    else:
        longest_major_axis = 0
    return longest_major_axis


def predict_from_model(patch, model):
    """Predict which pixels are tumor.
    
    input: patch: 256x256x3, rgb image
    input: model: keras model
    output: prediction: 256x256x1, per-pixel tumor probability
    """
    
    prediction = model.predict(patch.reshape(1, 256, 256, 3))
    prediction = prediction.reshape(256, 256)
    return prediction

def make_directory(DIR):
    if not os.path.isdir(DIR):
        os.mkdir(DIR)
        print(DIR,'made!')

def acc_score(truth, pred):
    cnt = 0

    for i in range(len(truth)):
        diff = np.abs(truth[i] - pred[i])
        if diff <= truth[i]*0.05 :
            cnt += 1
    return cnt / len(truth)