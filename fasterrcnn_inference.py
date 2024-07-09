# encoding:utf-8
import os
import json

import cv2
import tqdm
import torch
import numpy as np
from torchvision import transforms

device = "cuda:2"
ckpt_path = "./ckpt/voc_fasterrcnn.pt"
anno_json = "./preds/VOC2007Test/instances_test2007.json"
image_root_dir = "datasets/VOC07_12/test2007"


def preprocess(image_path):
    assert os.path.isfile(image_path), f"{image_path} not exists."
    image_org = cv2.imread(image_path)
    image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image_org)
    return image


def inference(image, model):
    """
    Args:
        image:
        model:
    Returns:
        a tuple (bboxes, labels, scores)
        bboxes:
            [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        labels:
            [label1, label2, ...]
        scores:
            [score1, score2, ...]  # 降序排列
    """
    image = image.unsqueeze(0).to(device)
    image = list(im.to(device) for im in image)
    batch_bboxes = []
    batch_labels = []
    batch_scores = []
    with torch.no_grad():
        outputs = model(image)
    for b in range(len(outputs)):
        output = outputs[b]
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        labels = labels.detach().cpu().numpy()
        batch_bboxes.append(boxes.detach().cpu().numpy())
        batch_labels.append(labels)
        batch_scores.append(scores.detach().cpu().numpy())
    bboxes = batch_bboxes[0]
    labels = batch_labels[0]
    scores = batch_scores[0]
    return bboxes, labels, scores


#  置信度阈值0.05，NMS阈值0.5
def postprocess(bboxes, labels, scores, conf_thresh=0.05, nms_thresh=0.5):
    keep = scores > conf_thresh
    bboxes = bboxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    return bboxes, labels, scores


def convert_onnx(model):
    model.eval()
    model.to("cpu")
    dummy_input = torch.randn(1, 3, 800, 1333).to("cpu")
    model(dummy_input) 
    im = torch.zeros(1, 3, 800, 1333).to("cpu") 
    torch.onnx.export(model, im, "./ckpt/faster_rcnn.onnx",
                      verbose=False,
                      opset_version=13,
                      training=torch.onnx.TrainingMode.EVAL,                     
                      do_constant_folding=True, 
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}})


if __name__ == "__main__":
    with open(anno_json, "r") as f:
        anno = json.load(f)
    images = anno["images"]
    preds = []
    model = torch.load(ckpt_path, map_location=device)
    convert_onnx(model)
    model.eval()
    for image_info in tqdm.tqdm(images):
        image_name = image_info["file_name"]
        image_id = image_info["id"]
        image_path = os.path.join(image_root_dir, image_name)

        image = preprocess(image_path)
        bboxes, labels, scores = inference(image, model)
        bboxes, labels, scores = postprocess(
            bboxes, labels, scores, conf_thresh=0.05, nms_thresh=0.5
        )

        for bbox, label, score in zip(bboxes, labels, scores):
            bbox = [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2] - bbox[0]),
                float(bbox[3] - bbox[1]),
            ]
            preds.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
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
        print("AP50:{}", map50)
        print("AP75:{}", map75)
    except Exception as e:
        print(f"pycocotools unable to run: {e}")
