from typing import Any, Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from mscv import load_checkpoint, save_checkpoint, load_state_dict

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.utils import load_state_dict_from_url

from .misc_nn_ops import Conv2dNormActivation
from .generalized_rcnn import GeneralizedRCNN
from .rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from .roi_heads import RoIHeads
from .transform import GeneralizedRCNNTransform
from .backbone_utils import resnet_fpn_backbone
from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool


__all__ = [
    "FasterRCNN", "fasterrcnn_resnet50_fpn","fasterrcnn_resnet101_fpn","fasterrcnn_resnet50_fpn_v2","fasterrcnn_vitbasep16_fpn",
    "fasterrcnn_vitlargep16_fpn"
]

def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)

class FasterRCNN(GeneralizedRCNN):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FasterRCNN
        >>> from torchvision.models.detection.rpn import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # FasterRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be [0]. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> # put the pieces together inside a FasterRCNN model
        >>> model = FasterRCNN(backbone,
        >>>                    num_classes=2,
        >>>                    rpn_anchor_generator=anchor_generator,
        >>>                    box_roi_pool=roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=0)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class FastRCNNConvFCHead(nn.Sequential):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        conv_layers: List[int],
        fc_layers: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        """
        Args:
            input_size (Tuple[int, int, int]): the input size in CHW format.
            conv_layers (list): feature dimensions of each Convolution layer
            fc_layers (list): feature dimensions of each FCN layer
            norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
        """
        in_channels, in_height, in_width = input_size

        blocks = []
        previous_channels = in_channels
        for current_channels in conv_layers:
            blocks.append(Conv2dNormActivation(previous_channels, current_channels, norm_layer=norm_layer))
            previous_channels = current_channels
        blocks.append(nn.Flatten())
        previous_channels = previous_channels * in_height * in_width
        for current_channels in fc_layers:
            blocks.append(nn.Linear(previous_channels, current_channels))
            blocks.append(nn.ReLU(inplace=True))
            previous_channels = current_channels

        super().__init__(*blocks)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}


def fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                            **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values between
          ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    if pretrained:
        # no need to download the backbone if pretrained is set
        # 如果Faster-RCNN整个网络加载权重就不需要backnone的权重
        pretrained_backbone = False

    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, norm_layer=norm_layer)

    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def fasterrcnn_resnet50_fpn_v2(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True,
                            **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        # 如果Faster-RCNN整个网络加载权重就不需要backnone的权重
        pretrained_backbone = False

    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, norm_layer=nn.BatchNorm2d)

    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], 
                       conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    model = FasterRCNN(
        backbone, 
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head = box_head,
        **kwargs,
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_v2_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def fasterrcnn_resnet101_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                            **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values between
          ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    if pretrained:
        # no need to download the backbone if pretrained is set
        # 如果Faster-RCNN整个网络加载权重就不需要backnone的权重
        pretrained_backbone = False

    backbone = resnet_fpn_backbone('resnet101', pretrained_backbone, norm_layer=norm_layer)

    model = FasterRCNN(backbone, num_classes, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model

def fasterrcnn_vitbasep16_fpn(pretrained=False, use_official_ckpt=True,
                            num_classes=91, pretrained_backbone=True,
                            **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        # 如果Faster-RCNN整个网络加载权重就不需要backnone的权重
        pretrained_backbone = False
    embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
    vit = ViT(img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",)
    if pretrained_backbone and use_official_ckpt:
        print('Loading MAE Pretrained ViT Base weights...')
        ckpt = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',map_location='cpu')
        vit.load_state_dict(ckpt['model'],strict=False)

    backbone = SimpleFeaturePyramid(
        vit,
        in_feature="last_feat",
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=LastLevelMaxPool(),
        norm="LN",
        square_pad=1024,
    )
    backbone.out_channels = 256
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=backbone._out_features,
        output_size=7,
        sampling_ratio=2
    )
    model = FasterRCNN(
        backbone, 
        num_classes=num_classes,
        box_roi_pool=roi_pooler,
    )
    return model

def fasterrcnn_vitlargep16_fpn(pretrained=False, use_official_ckpt=True,
                            num_classes=91, pretrained_backbone=True,
                            **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        # 如果Faster-RCNN整个网络加载权重就不需要backnone的权重
        pretrained_backbone = False
    embed_dim, depth, num_heads, dp = 1024, 24, 16, 0.4
    vit = ViT(img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # 5, 11, 17, 23 for global attention
        window_block_indexes=list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23)),
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",)
    
    if pretrained_backbone:
        if use_official_ckpt:
            print('Loading Official MAE Pretrained ViT Large weights...')
            ckpt = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',map_location='cpu')
        else:
            print('Loading RETFound MAE Pretrained ViT Large weights...',map_location='cpu')
            ckpt = torch.load("/data/hanliming/checkpoints/RETFound_ckpt/RETFound_oct_weights.pth")
        vit.load_state_dict(ckpt["model"], strict=False)
    backbone = SimpleFeaturePyramid(
        vit,
        in_feature="last_feat",
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=LastLevelMaxPool(),
        norm="LN",
        square_pad=1024,
    )
    backbone.out_channels = 256
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=backbone._out_features,
        output_size=7,
        sampling_ratio=2
    )
    model = FasterRCNN(
        backbone, 
        num_classes=num_classes,
        box_roi_pool=roi_pooler,
    )
    return model