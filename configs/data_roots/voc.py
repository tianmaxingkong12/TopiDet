import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class Data(object):
    data_format = 'VOC'
    voc_root = 'datasets/VOC07_12'
    train_split = 'train0712.txt'
    val_split = 'val0712.txt'
    test_split = 'test07.txt'
    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    img_format = 'jpg'

