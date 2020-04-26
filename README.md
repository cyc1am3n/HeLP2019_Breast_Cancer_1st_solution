# HeLP Challenge 2019 Breast Cancer 1st place solution

This repository is **1st place solution** to the **Breast Cancer Classification Task of HeLP Challenge 2019**.  
![task_description](./assets/task_description.png)


## Model
![model_description](./assets/model_description.png)
### Stage 1
- Preprocessing: ROI extraction, Rescale, Vahadane Stain Normalization
- Pixel-wise Segmentation: Feature Pyramid Network(FPN)
### Stage 2
- Feature extraction from probability heatmap
- Prediction final probability and major axis based on features

And also, please click [this link](./assets/slide.pdf) to see the detailed model description.

## Dependencies
- keras
- segmentation_models
- openslide
- staintools
- numpy
- pandas
- sklearn
- skimage

## Usage

### Dataset

```bash
data
  └── train
     ├── level4
     │  ├── Image
     │  │  ├── slide_001.png
     │  │  ├── ...
     │  │  └── slide_#.png
     │  └── Mask
     │     ├── mask_001.png
     │	   ├── ...
     │	   └── mask_#.png
     └── label.csv
            
========= After training, the directories are created as below. =========

  ├── volume
  │  ├── dataset
  │  │  └── level4 
  │  │     ├── img
  │  │	   │  ├── slide001_patch001.png
  │  │ 	   │  ├── ...
  │  │     │  └── slide#_patch#.png
  │  │	   └── mask
  │  │	      ├── mask001_patch001.png
  │  │        ├── ...
  │  │        └── mask#_patch#.png
  │  └── model
  │       └── fpn_weights.h5
  └── heatmap
      ...
```



### Train
Run the `train.py`.  
```bash
$ python train.py
```
### Inference
Run the `inference.sh`.
```bash
$ sh inference.sh
```

## Authors
- Daeyoung Kim / [@cyc1am3n](https://github.com/cyc1am3n)  
- Taewoo Kim / [@Taeu](https://github.com/Taeu)  
- Jonghyun Choi / [@ExcelsiorCJH](https://github.com/ExcelsiorCJH)
