import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# COCO 2017 val in COCO format with only voc classes
class Data(object):
    data_format = 'COCO'
    voc_root = 'datasets/COCO2017'
    train_split = 'val2017'
    val_split = 'val2017'
    test_split = "val2017"
    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
