import random
import os
import json

from pycocotools.coco import COCO

random.seed(42)


def print_coco(json_path):
    coco = COCO(annotation_file=json_path)
    print("COCO数据集类别:{}".format(len(coco.cats)))
    print("COCO图像数目:{}".format(len(coco.imgs)))
    print(
        "{0:<5} {1:<5} {2:<5} {3:<5}".format(
            "类别ID", "类别名称", "图片数目", "目标数目"
        )
    )
    cat_ids = coco.getCatIds()
    for cat_id in cat_ids:
        _id = cat_id
        _cat = coco.cats[_id]["name"]
        _images = coco.getImgIds(catIds=_id)
        _annos = coco.getAnnIds(catIds=_id)
        print(
            "{0:<5},{1:<15},{2:<5},{3:<5}".format(_id, _cat, len(_images), len(_annos))
        )


if __name__ == "__main__":
    ## 1.划分训练集 抽取5k张图片
    val_num = 5000
    train_json_path = "./datasets/COCO2017/annotations/instances_train2017.json"
    output_dir = "./datasets/COCO2017/annotations/"
    coco = COCO(annotation_file=train_json_path)
    print("COCO数据集类别:{}".format(len(coco.cats)))
    print("COCO训练集图像数目:{}".format(len(coco.imgs)))
    with open(train_json_path, "r") as f:
        coco_json = json.load(f)
    imageIds = coco.getImgIds()
    val_imageIds = set(random.sample(imageIds, val_num))
    train_coco = dict()
    val_coco = dict()
    for _ in ["train", "val"]:
        _coco = dict()
        _coco["info"] = coco_json["info"]
        _coco["licenses"] = coco_json["licenses"]
        _coco["categories"] = coco_json["categories"]
        images = []
        annos = []
        for image in coco_json["images"]:
            if _ == "train" and image["id"] not in val_imageIds:
                images.append(image)
            elif _ == "val" and image["id"] in val_imageIds:
                images.append(image)
        for anno in coco_json["annotations"]:
            if _ == "train" and anno["image_id"] not in val_imageIds:
                annos.append(anno)
            elif _ == "val" and anno["image_id"] in val_imageIds:
                annos.append(anno)
        _coco["images"] = images
        _coco["annotations"] = annos
        save_path = os.path.join(output_dir, "instances_" + _ + "0511.json")
        with open(save_path, "w") as f:
            json.dump(_coco, f)
        print_coco(save_path)