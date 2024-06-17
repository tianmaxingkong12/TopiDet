import random
import os

from pycocotools.coco import COCO

voc_classes = [
    "airplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorcycle",
    "person",
    "potted plant",
    "sheep",
    "couch",
    "train",
    "tv",
]

if __name__ == "__main__":
    ## 1. 统计包含VOC类别的图片数目及目标数目
    val_json_path = "./datasets/COCO2017/annotations/instances_val2017.json"
    train_json_path = "./datasets/COCO2017/annotations/instances_train2017.json"
    for json_path in [val_json_path,train_json_path]:
        coco = COCO(annotation_file=json_path)
        print("COCO数据集类别:{}".format(len(coco.cats)))
        print("COCO图像数目:{}".format(len(coco.imgs)))
        print(
            "{0:<5} {1:<5} {2:<5} {3:<5}".format(
                "类别ID", "类别名称", "图片数目", "目标数目"
            )
        )

        image_ids = []
        anno_ids = coco.getAnnIds(catIds=coco.getCatIds(catNms=voc_classes))
        for _ in voc_classes:
            _cat = _
            _id = coco.getCatIds(catNms=[_cat])[0]
            _images = coco.getImgIds(catIds=[_id])
            _annos = coco.getAnnIds(catIds=[_id])
            image_ids.extend(_images)
            print(
                "{0:<5},{1:<15},{2:<5},{3:<5}".format(_id, _cat, len(_images), len(_annos))
            )
        image_ids = list(set(image_ids))
        print(
            "COCO2017包含VOC类别的图像总数:{}，目标总数:{}".format(
                len(image_ids),
                len(anno_ids),
            )
        )
