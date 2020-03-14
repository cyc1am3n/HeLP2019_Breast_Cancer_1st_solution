import segmentation_models as sm
from keras.optimizers import Adam

"""
Network Architecture
"""

def fpn(backbone, pretrained_weights=None):
    model = sm.FPN(backbone, 
                   input_shape=(256, 256, 3), 
                   classes=1, 
                   activation='sigmoid', 
                   encoder_weights=pretrained_weights)
    
    model.compile(optimizer='adam', 
                  loss=sm.losses.bce_jaccard_loss, 
                  metrics=[sm.metrics.iou_score, sm.metrics.f1_score])
    return model


def unet(backbone, pretrained_weights=None):
    model = sm.Unet(backbone, 
                    input_shape=(256, 256, 3), 
                    classes=1, 
                    activation='sigmoid',
                    encoder_weights=pretrained_weights)

    model.compile(optimizer='adam', 
                  loss=sm.losses.bce_jaccard_loss, 
                  metrics=[sm.metrics.iou_score, sm.metrics.f1_score])
    
    return model