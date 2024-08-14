import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt_path = "./preds/VOC2007Test/instances_test2007.json"
pred_path = "./preds/VOC2007Test/pred_fasterrcnn_cocopretrained1.json"
save_root = "./preds/VOC2007Test/pred_fasterrcnn_cocopretrained1/"
if os.path.exists(save_root) == False:
    os.makedirs(save_root,exist_ok=True)

template = {
    "isDifficult": "0",
    "picQuality": "1",
    "markList": [
        {
            "code": "bus",
            "color": "rgb(153,255,153)",
            "dataList": [
                {
                    "endY": "228",
                    "endX": "302",
                    "occluded": "0",
                    "difficult": "0",
                    "startY": "163",
                    "startX": "72",
                    "isTruncation": "0",
                }
            ],
            "name": "bus",
            "type": 4,
        },
    ],
    "project": "通用画框识别",
    "version": "V1.0",
    "picName": "VOC2007_test_000014",
}


def count(fp):
    with open(fp, "r") as f:
        data = json.load(f)
    print(len(data))


def save_txt(save_path, dic):
    with open(save_path, "w", encoding='utf-8') as fp:
        fp.write(json.dumps(dic,ensure_ascii=False))


def convert_single_class_anno(annos, cls_name):
    anno = {}
    anno["code"] = cls_name
    anno["name"] = cls_name
    anno["type"] = 4
    dataList = []
    for _ in annos:
        _convert_annos = {}
        # _convert_annos["startX"] = int(_["bbox"][0])
        # _convert_annos["startY"] = int(_["bbox"][1])
        # _convert_annos["endX"] = int(_["bbox"][0] + _["bbox"][2])
        # _convert_annos["endY"] = int(_["bbox"][1] + _["bbox"][3])
        _convert_annos["startX"] = _["bbox"][0]
        _convert_annos["startY"] = _["bbox"][1]
        _convert_annos["endX"] = _["bbox"][0] + _["bbox"][2]
        _convert_annos["endY"] = _["bbox"][1] + _["bbox"][3]
        _convert_annos["confidence"] = _["score"]
        dataList.append(_convert_annos)
    anno["dataList"] = dataList
    return anno


if __name__ == "__main__":
    count(pred_path)
    cocoGT = COCO(gt_path)
    cocoPred = cocoGT.loadRes(pred_path)
    gt_image_ids = cocoGT.getImgIds()
    pred_image_ids = cocoPred.getImgIds()
    pred_anno_ids = cocoPred.getAnnIds()
    cat_ids = cocoPred.getCatIds()
    for i in pred_image_ids:
        markList = []
        for j in cat_ids:
            _anno_ids = cocoPred.getAnnIds(imgIds=[i], catIds=[j])
            if len(_anno_ids):
                _annos = cocoPred.loadAnns(_anno_ids)
                markList.append(
                    convert_single_class_anno(_annos, cocoPred.cats[j]["name"])
                )
        template["markList"] = markList
        image_name = "VOC2007_test_" + cocoPred.imgs[i]["file_name"][:-4]
        template["picName"] = image_name
        save_txt(os.path.join(save_root, image_name + ".txt"), template)