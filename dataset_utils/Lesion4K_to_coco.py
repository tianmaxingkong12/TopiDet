# encoding=utf-8
import os
import json
import random
import sys
import cv2
import misc_utils as utils
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--anno_dir', type=str, default=None,
                        help='anno dir', required=True)
    parser.add_argument('--image_dir', type=str, default=None,
                        help='image dir', required=True)
    parser.add_argument('--save_anno_path', type=str, default=None,
                        help='save anno path',required=True)
    
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()

    coco_dataset = {'info':{}, 'images': [], 'annotations': [], 'categories': [], 'licenses': []}
    global_image_id = 0
    global_annotation_id = 0

    coco_dataset['info']["description"] = "Lesion-4K"
    coco_dataset['info']["url"] = ""
    coco_dataset['info']["version"] = "V2.0.5"
    coco_dataset['info']["year"] = "2024"
    coco_dataset['info']["contributor"] = ""
    coco_dataset["info"]["data_created"] = "2024/06/17"

    lesions = [(37,"PVD"),(41,"RD"),(8,"IRF"),(21,"后巩膜葡萄肿"),(1,"ERM"),(42,"NsPED"),(11,"HRD"),
               (30,"RPE萎缩"),(20,"双层征"),(2,"MH"),(16,"sPED"),(25,"视盘水肿"),(40,"Drusen"),(19,"CHD曲度异常"),
               (44,"视网膜增殖膜"),(3,"VMT")]

    for lesion in lesions:
        coco_dataset['categories'].append({   
            'supercategory': 'Lesion',
            'id': lesion[0],
            'name': lesion[1]
        })
    for anno_name in os.listdir(opt.anno_dir):
        if anno_name.endswith(".xlsx"):
            continue
        image_name = anno_name.replace(".txt",".jpg")
        image = cv2.imread(os.path.join(opt.image_dir,image_name))
        h, w, _ = image.shape
        del image

        with open(os.path.join(opt.anno_dir,anno_name), 'r',encoding='utf-8') as f:
            content = f.readlines()[0]
        anno = json.loads(content)
        assert anno["picName"] == image_name[:-4]
        coco_dataset['images'].append({
            'license': "",
            'file_name': image_name,
            'coco_url': "",
            'id': global_image_id,
            'width': int(w),
            'height': int(h),
            'date_captured': "",
            'flickl_url': "",
            'isdifficult': int(anno["isDifficult"]),
            'effectiveAreaLeft': anno["effectiveAreaLeft"],
            'effectiveAreaRight': anno["effectiveAreaRight"]
            })
        
        for lesion_mark in anno["markList"]:
            if lesion_mark["code"] == "0":
                continue
            for mark in lesion_mark["dataList"]:
                x1, y1, x2, y2 = mark["startX"],mark["startY"],mark["endX"],mark["endY"]
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2),  
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                area = width * height
                coco_dataset['annotations'].append({
                    'id': global_annotation_id,
                    'image_id': global_image_id,
                    'category_id': int(lesion_mark["code"]),
                    'segmentation': [[]],
                    'area': area,
                    'bbox': [x1, y1, width, height],
                    'iscrowd': 0,
                    'istypical': int(mark["isTypical"]),
                    # 'istruncation': int(mark["isTruncation"])
                })
                global_annotation_id += 1
        global_image_id += 1

    with open(opt.save_anno_path, 'w', encoding='utf-8') as f:
        json.dump(coco_dataset, f, ensure_ascii=False)

    print(f'Done. coco json file has been saved to `{opt.save_anno_path}`')