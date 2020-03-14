import os
import sys
sys.path.append('..')

def make_directory():
    if not os.path.exists('../data/volume/train'):
    os.mkdir('../data/volume/train')
    if not os.path.exists('../data/volume/valid'):
        os.mkdir('../data/volume/valid')

    if not os.path.exists('../data/volume/train/level4'):
        os.mkdir('../data/volume/train/level4')
    if not os.path.exists('../data/volume/valid/level4'):
        os.mkdir('../data/volume/valid/level4')   

    if not os.path.exists('../data/volume/train/level4/img'):
        os.mkdir('../data/volume/train/level4/img')
    if not os.path.exists('../data/volume/train/level4/mask'):
        os.mkdir('../data/volume/train/level4/mask')
        
    if not os.path.exists('../data/volume/valid/level4/img'):
        os.mkdir('../data/volume/valid/level4/img')
    if not os.path.exists('../data/volume/valid/level4/mask'):
        os.mkdir('../data/volume/valid/level4/mask')

    # 0: normal, 1: tumor
    if not os.path.exists('../data/volume/train/level4/img/0'):
        os.mkdir('../data/volume/train/level4/img/0')
    if not os.path.exists('../data/volume/train/level4/mask/0'):
        os.mkdir('../data/volume/train/level4/mask/0')
    if not os.path.exists('../data/volume/train/level4/img/1'):
        os.mkdir('../data/volume/train/level4/img/1')
    if not os.path.exists('../data/volume/train/level4/mask/1'):
        os.mkdir('../data/volume/train/level4/mask/1')
    # 0: normal, 1: tumor
    if not os.path.exists('../data/volume/valid/level4/img/0'):
        os.mkdir('../data/volume/valid/level4/img/0')
    if not os.path.exists('../data/volume/valid/level4/mask/0'):
        os.mkdir('../data/volume/valid/level4/mask/0')
    if not os.path.exists('../data/volume/valid/level4/img/1'):
        os.mkdir('../data/volume/valid/level4/img/1')
    if not os.path.exists('../data/volume/valid/level4/mask/1'):
        os.mkdir('../data/volume/valid/level4/mask/1')

    print('Created Directories')
    return None