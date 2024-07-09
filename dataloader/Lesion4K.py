import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from options import opt
from dataset_utils.Lesion_COCO import Lesion_COCO
import cv2

from dataloader.data_helper import voc_to_yolo_format

def convert_index_to_Leison_label(id):
    m = [1,2,3,8,11,16,19,20,21,25,30,37,40,41,42,44]
    return m[id]

class Lesion4KDataset(Dataset):
    """Coco dataset."""

    def __init__(
        self,
        root_dir,
        set_name,
        transforms=None,
        load_difficult=True,
        load_nontypicalbbox=True,
        load_effectivearea=False,
    ):
        """
        Args:
            root_dir (string): Lesion-4K directory.
            transforms (callable, optional): Optional transforms to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transforms = transforms
        self.load_difficult = load_difficult
        self.load_nontypicalbbox = load_nontypicalbbox
        self.load_effectivearea = load_effectivearea

        self.coco = Lesion_COCO(
            os.path.join(
                self.root_dir, "annotations", "instances_" + self.set_name + ".json"
            )
        )
        self.image_ids = self.coco.getImgIds(contain_difficult=self.load_difficult)
        ## TODO 可配置是否加载其他类别 （Drusen, 视盘水肿，CHD曲度异常，VMT, 视网膜增殖膜）
        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.classes = {}
        self.lesion_labels = {}
        self.lesion_labels_inverse = {}
        for c in categories:
            self.lesion_labels[len(self.classes)] = c["id"] #按照病灶编码顺序从0开始递增
            self.lesion_labels_inverse[c["id"]] = len(self.classes)
            self.classes[c["name"]] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img, path = self.load_image(idx)
        annot = self.load_annotations(idx)
        bboxes = annot[:, :4]
        labels = annot[:, 4]

        if self.transforms:
            num_transforms = 20 if len(bboxes) else 1
            for i in range(num_transforms):
                sample = self.transforms(
                    **{"image": img, "bboxes": bboxes, "labels": labels}
                )

                if len(sample["bboxes"]) > 0:
                    break

        sample["bboxes"] = torch.Tensor(sample["bboxes"])
        sample["labels"] = torch.Tensor(sample["labels"])
        sample["path"] = path
        sample["image_id"] = self.image_ids[idx]

        sample.update(voc_to_yolo_format(sample))  # yolo format

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]

        path = os.path.join(self.root_dir, self.set_name, image_info["file_name"])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #
        # if len(img.shape) == 2:
        #     img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0, path

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False, istypical=not self.load_nontypicalbbox
        )
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a["bbox"][2] < 1 or a["bbox"][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a["bbox"]
            annotation[0, 4] = self.coco_label_to_label(a["category_id"])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.lesion_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.lesion_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image["width"]) / float(image["height"])

    def num_classes(self):
        return 16