import pdb
import sys

import numpy as np
import torch
import cv2
import os
from collections import OrderedDict,defaultdict
from torch import nn

from options import opt
from options.helper import is_distributed, is_first_gpu

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init
from mscv.summary import write_image

from utils import to_2tuple
import misc_utils as utils
import ipdb

from .frcnn.faster_rcnn import FasterRCNN, FastRCNNPredictor
from .frcnn import fasterrcnn_resnet50_fpn
from .frcnn.rpn import concat_box_prediction_layers
from .frcnn.roi_heads import fastrcnn_loss

# from dataloader.coco import coco_90_to_80_classes

from .backbones import vgg16_backbone, res101_backbone


class Model(BaseModel):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, kwargs)
        self.config = config

        kargs = {}
        if 'SCALE' in config.DATA:
            scale = config.DATA.SCALE
            if isinstance(scale, int):
                min_size = scale
                max_size = int(min_size / 3 * 5)
            else:
                min_size, max_size = config.DATA.SCALE

            kargs = {'min_size': min_size,
                     'max_size': max_size,
                    }
        
        kargs.update({'box_nms_thresh': config.TEST.NMS_THRESH})

        # 多卡使用 SyncBN
        if is_distributed():
            kargs.update({'norm_layer': torch.nn.SyncBatchNorm})

        # 定义backbone和Faster RCNN模型
        if config.MODEL.BACKBONE is None or config.MODEL.BACKBONE.lower() in ['res50', 'resnet50']:
            # 默认是带fpn的resnet50
            self.detector = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=config.MODEL.BACKBONE_PRETRAINED, **kargs)

            in_features = self.detector.roi_heads.box_predictor.cls_score.in_features

            # replace the pre-trained head with a new one
            self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.DATA.NUM_CLASSESS + 1)

        elif config.MODEL.BACKBONE.lower() in ['vgg16', 'vgg']:
            backbone = vgg16_backbone(config.MODEL.BACKBONE_PRETRAINED)
            self.detector = FasterRCNN(backbone, num_classes=config.DATA.NUM_CLASSESS + 1, **kargs)

        elif config.MODEL.BACKBONE.lower() in ['res101', 'resnet101']:
            # 不带FPN的resnet101
            backbone = res101_backbone(config.MODEL.BACKBONE_PRETRAINED)
            self.detector = FasterRCNN(backbone, num_classes=config.DATA.NUM_CLASSESS + 1, **kargs)

        elif config.MODEL.BACKBONE.lower() in ['res', 'resnet']:
            raise RuntimeError(f'backbone "{config.MODEL.BACKBONE}" is ambiguous, please specify layers.')

        else:
            raise NotImplementedError(f'no such backbone: {config.MODEL.BACKBONE}')


        if opt.debug and is_first_gpu():
            print_network(self.detector)

        self.to(opt.device)
        # 多GPU支持
        if is_distributed():
            self.detector = torch.nn.parallel.DistributedDataParallel(self.detector, find_unused_parameters=False,
                    device_ids=[opt.local_rank], output_device=opt.local_rank)
            # self.detector = torch.nn.parallel.DistributedDataParallel(self.detector, device_ids=[opt.local_rank], output_device=opt.local_rank)

        self.optimizer = get_optimizer(config, self.detector)
        self.scheduler = get_scheduler(config, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.loss_details = dict()


    def update(self, sample, *arg):
        """
        给定一个batch的图像和gt, 更新网络权重, 仅在训练时使用.
        Args:
            sample: {'input': a Tensor [b, 3, height, width],
                   'bboxes': a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]],
                   'labels': a list of labels [[N1], [N2], ..., [Nb]],
                   'path': a list of paths}
        """
        labels = sample['labels']
        for label in labels:
            label += 1.  # effdet的label从1开始

        image, bboxes, labels = sample['image'], sample['bboxes'], sample['labels']
        
        for b in range(len(image)):
            if len(bboxes[b]) == 0:  # 没有bbox，不更新参数
                return {}

        #image = image.to(opt.device)
        bboxes = [bbox.to(opt.device).float() for bbox in bboxes]
        labels = [label.to(opt.device).float() for label in labels]
        image = list(im.to(opt.device) for im in image)

        b = len(bboxes)

        target = [{'boxes': bboxes[i], 'labels': labels[i].long()} for i in range(b)]
        """
            target['boxes'] = boxes
            target['labels'] = labels
            # target['masks'] = None
            target['image_id'] = torch.tensor([index])
            target['area'] = area
            target['iscrowd'] = iscrowd
        """
        loss_dict = self.detector(image, target) ##返回值包含损失和检测框

        loss = sum(l for l in loss_dict.values())

        self.avg_meters.update({'loss': loss.item()})

        ## 添加需要输出的loss信息
        for _loss in loss_dict:
            self.loss_details["train/"+_loss] = loss_dict[_loss].item()
        self.loss_details["train/"+"loss"] = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}
    
    def valid(self, dataloader):
        if is_distributed():
            detector = self.detector.module
        loss_details = defaultdict(float)
        bs = 0
        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                labels = sample['labels']
                for label in labels:
                    label += 1.  # effdet的label从1开始
                image, bboxes, labels = sample['image'], sample['bboxes'], sample['labels']
                for b in range(len(image)):
                    if len(bboxes[b]) == 0:  # 没有bbox，不更新参数
                        return {}
                bboxes = [bbox.to(opt.device).float() for bbox in bboxes]
                labels = [label.to(opt.device).float() for label in labels]
                images = list(im.to(opt.device) for im in image)
                b = len(bboxes)
                targets = [{'boxes': bboxes[i], 'labels': labels[i].long()} for i in range(b)]

                original_image_sizes = [img.shape[-2:] for img in images]
                images, targets = detector.transform(images, targets)
                features = detector.backbone(images.tensors)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([(0, features)])
                # RPN网络
                # RPN uses all feature maps that are available
                features1 = list(features.values())
                objectness, pred_bbox_deltas = detector.rpn.head(features1)
                anchors = detector.rpn.anchor_generator(images, features1)
                num_images = len(anchors)
                num_anchors_per_level = [o[0].numel() for o in objectness]
                objectness, pred_bbox_deltas = \
                        concat_box_prediction_layers(objectness, pred_bbox_deltas)
                # apply pred_bbox_deltas to anchors to obtain the decoded proposals
                # note that we detach the deltas because Faster R-CNN do not backprop through
                # the proposals
                proposals = detector.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
                proposals = proposals.view(num_images, -1, 4)
                boxes, scores = detector.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
                labels, matched_gt_boxes = detector.rpn.assign_targets_to_anchors(anchors, targets)
                regression_targets = detector.rpn.box_coder.encode(matched_gt_boxes, anchors)
                loss_objectness, loss_rpn_box_reg = detector.rpn.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
                RPN_losses = {
                        "loss_objectness": loss_objectness,
                        "loss_rpn_box_reg": loss_rpn_box_reg,
                    }

                # ROIhead
                proposals, matched_idxs, labels, regression_targets = detector.roi_heads.select_training_samples(proposals, targets)
                image_shapes = images.image_sizes
                box_features = detector.roi_heads.box_roi_pool(features, proposals, image_shapes)
                box_features = detector.roi_heads.box_head(box_features)
                class_logits, box_regression = detector.roi_heads.box_predictor(box_features)
                loss_classifier, loss_box_reg = fastrcnn_loss(
                    class_logits, box_regression, labels, regression_targets)
                detector_losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

                loss_dict = {}
                loss_dict.update(RPN_losses)
                loss_dict.update(detector_losses)

                for _loss in loss_dict:
                    loss_details["val/"+_loss] += loss_dict[_loss].item()
                loss_details["val/"+"loss"] += sum(l for l in loss_dict.values()).item()
                bs += 1
        for _loss in loss_details:
            loss_details[_loss] /= bs
        return loss_details
        

    def forward_test(self, image):  # test
        """给定一个batch的图像, 输出预测的[bounding boxes, labels和scores], 仅在验证和测试时使用"""
        #image = list(im for im in image)
        image = list(im.to(opt.device) for im in image)

        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        with torch.no_grad():
            outputs = self.detector(image)

        conf_thresh = self.config.TEST.CONF_THRESH

        for b in range(len(outputs)):  #
            output = outputs[b]
            boxes = output['boxes']
            labels = output['labels']
            scores = output['scores']
            boxes = boxes[scores > conf_thresh]
            labels = labels[scores > conf_thresh]
            labels = labels.detach().cpu().numpy()
            # for i in range(len(labels)):
            #     labels[i] = coco_90_to_80_classes(labels[i])

            labels = labels - 1
            scores = scores[scores > conf_thresh]

            batch_bboxes.append(boxes.detach().cpu().numpy())
            batch_labels.append(labels)
            batch_scores.append(scores.detach().cpu().numpy())

        return batch_bboxes, batch_labels, batch_scores

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, save_dir, which_epoch):
        super(Model, self).save(save_dir,which_epoch)
