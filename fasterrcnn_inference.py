# encoding:utf-8
import os
import json

import cv2
import tqdm
import torch
import numpy as np
from torchvision import transforms
from torchvision.ops import boxes as box_ops

def coco80_to_voc20_class():
    m = [-1]*80
    m[4] = 1
    m[1] = 2
    m[14] = 3
    m[8] = 4
    m[39] = 5
    m[5] = 6
    m[2] = 7
    m[15] = 8
    m[56] = 9
    m[19] = 10
    m[60] = 11
    m[16] = 12
    m[17] = 13
    m[3] = 14
    m[0] = 15 #person
    m[58] = 16
    m[18] = 17
    m[57] = 18
    m[6] = 19
    m[62] = 20
    return m
def coco90_to_coco80_class():
    m = [0,1,2,3,4,5,6,7,8,9,10,11,0,12,13,14,15,16,17,18,19,20,21,22,23,24,0,
            25,26,0,0,27,28,29,30,31,32,33,34,35,36,37,38,39,40, 0,41,42,43,44,45,46,
            47,48,49,50,51,52,53,54,55,56,57,58,59,60, 0,61,0,0,62,0,63,64,65,66,67,68,69,
            70,71,72,73,0,74,75,76,77,78,79,80]
    return m
def coco90_to_voc20_class():
    m = []
    coco90_to_80 = coco90_to_coco80_class()
    coco80_to_voc20 = coco80_to_voc20_class()
    for _ in range(91): #1~90
        if coco90_to_80[_] == 0:
            m.append(-1)
        else:
            m.append(coco80_to_voc20[coco90_to_80[_]-1])
    return m
    

device = "cuda:2"
ckpt_path = "./ckpt/FasterRCNN-Official-COCO.pt"
anno_json = "./preds/VOC2007Test/instances_test2007.json"
image_root_dir = "datasets/VOC07_12/test2007"


def inference(image_path, model, conf_thresh=0.05, nms_thresh=0.5):
    assert os.path.isfile(image_path), f"{image_path} not exists."
    ## 前处理
    image_org = cv2.imread(image_path)
    image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(image_org)
    img = img.unsqueeze(0).to(device)
    
    ## 模型推理
    with torch.no_grad():
        outputs = model(img)
    predicition = outputs[0]

    ## 后处理
    boxes = np.array([])
    scores = np.array([])
    labels = np.array([])
    if predicition["boxes"].shape[0]:
        boxes, scores, labels = post_process(predicition, nms_thresh, conf_thresh)
        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().astype("int64")
    return boxes, labels, scores


#  置信度阈值0.05，NMS阈值0.5
def post_process(detection, nms_thresh=0.5, conf_thresh=0.05, max_dets=300):
    boxes = detection["boxes"]
    labels = detection["labels"]
    scores = detection["scores"]
    # remove low scoring boxes
    inds = torch.nonzero(scores > conf_thresh).squeeze(1)
    boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

    # remove empty boxes
    keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # # non-maximum suppression, independently done per class
    # keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)
    # # keep only topk scoring predictions
    # keep = keep[:max_dets]
    # boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    return boxes, scores, labels


def convert_onnx(model):
    model.eval()
    model.to("cpu")
    dummy_input = torch.randn(1, 3, 800, 1333).to("cpu")
    model(dummy_input) 
    im = torch.zeros(1, 3, 800, 1333).to("cpu") 
    torch.onnx.export(model, im, "./ckpt/FasterRCNN-Official-COCO.onnx",
                      verbose=False,
                      opset_version=13,               
                      do_constant_folding=True, 
                      input_names=['input'],
                      output_names=["boxes","labels","scores"],
                      dynamic_axes={"input": {0: "batch",2:"h",3:"w"}, "boxes": {0: "anchors"}, "scores":{0:"anchors"}, "labels":{0:"anchors"}})


if __name__ == "__main__":
    class_map = coco90_to_voc20_class()
    with open(anno_json, "r") as f:
        anno = json.load(f)
    images = anno["images"]
    preds = []
    model = torch.load(ckpt_path, map_location=device)
    # convert_onnx(model)
    model.eval()
    for image_info in tqdm.tqdm(images):
        image_name = image_info["file_name"]
        image_id = image_info["id"]
        image_path = os.path.join(image_root_dir, image_name)
        bboxes, labels, scores = inference(image_path, model, conf_thresh=0.05, nms_thresh=0.5)
        for bbox, label, score in zip(bboxes, labels, scores):
            if class_map[int(label)] != -1:
                bbox = [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]-bbox[0]),
                    float(bbox[3]-bbox[1]),
                ]
                preds.append(
                    {
                        "image_id": image_id,
                        "category_id": class_map[int(label)],
                        "bbox": bbox,
                        "score": float(score),
                    }
                )
    pred_json = os.path.join(".", "pred.json")
    with open(pred_json, "w") as f:
        json.dump(preds, f)

    ## coco evaluate
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, "bbox")
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50, map75 = eval.stats[:3]  # update results (mAP@0.5:0.95, mAP@0.5)
        print("AP50:{}".format(map50))
        print("AP75:{}".format(map75))
    except Exception as e:
        print(f"pycocotools unable to run: {e}")
