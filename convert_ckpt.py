# encoding:utf-8
import torch
# from options import opt, config
from torchvision.models.detection import fasterrcnn_resnet50_fpn

voc_class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

ckpt_path = './ckpt/13_Faster_RCNN.pt'
device = 'cuda:3'
INFERENCE_LIST = 'datasets/voc/ImageSets/Main/test.txt'
IMAGE_FOLDER = 'datasets/voc/JPEGImages'
KEEP_THRESH = 0.5
SAVE_PATH = 'results/inference'

# config.DATA.NUM_CLASSESS = len(voc_class_names)
model = fasterrcnn_resnet50_fpn(num_classes=21, box_score_thresh=0.05, box_nms_thresh=0.99, box_detections_per_img=1000)
model = model.to(device=device)
model.load_state_dict(torch.load(ckpt_path,map_location=device)["detector"])
torch.save(model,"./ckpt/FasterRCNN-HLM-VOC.pt")