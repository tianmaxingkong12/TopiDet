import json

source_anno_file = "./preds/VOC2007Test/pred_fasterrcnn_cocopretrained_raw.json"
target_anno_file = "./preds/VOC2007Test/pred_fasterrcnn_cocopretrained.json"
coco_to_voc = {
    5: 1,
    2: 2,
    16: 3,
    9: 4,
    44: 5,
    6: 6,
    3: 7,
    17: 8,
    62: 9,
    21: 10,
    67: 11,
    18: 12,
    19: 13,
    4: 14,
    1: 15,
    64: 16,
    20: 17,
    63: 18,
    7: 19,
    72: 20,
}

voc_to_coco = {
    1: 5,
    2: 2,
    3: 16,
    4: 9,
    5: 44,
    6: 6,
    7: 3,
    8: 17,
    9: 62,
    10: 21,
    11: 67,
    12: 18,
    13: 19,
    14: 4,
    15: 1,
    16: 64,
    17: 20,
    18: 63,
    19: 7,
    20: 72,
}

if __name__ == "__main__":
    with open(source_anno_file, "r") as f:
        data = json.load(f)
    new_data = []
    for _ in data:
        if _["category_id"] in coco_to_voc.keys():
            _["category_id"] = coco_to_voc[_["category_id"]]
            new_data.append(_)
    print(len(data),len(new_data))
    with open(target_anno_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False)
