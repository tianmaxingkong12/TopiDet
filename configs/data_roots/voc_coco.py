import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# VOC2007 Test in COCO format
class Data(object):
    data_format = 'COCO'
    voc_root = 'datasets/VOC07_12'
    train_split = 'test2007'
    val_split = 'test2007'
    test_split = "test2007"
    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
