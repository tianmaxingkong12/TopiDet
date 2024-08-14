import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import numpy as np

# from .Faster_RCNN.frcnn.misc_nn_ops import Conv2dNormActivation, MLP
import timm.models.vision_transformer
from detectron2.modeling.backbone import Backbone


__all__ = [
    "VisionTransformer",
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
    "interpolate_pos_embed",
]


# class ConvStemConfig(NamedTuple):
#     out_channels: int
#     kernel_size: int
#     stride: int
#     norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
#     activation_layer: Callable[..., nn.Module] = nn.ReLU


# class MLPBlock(MLP):
#     """Transformer MLP block."""

#     _version = 2

#     def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
#         super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.normal_(m.bias, std=1e-6)

#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     ):
#         version = local_metadata.get("version", None)

#         if version is None or version < 2:
#             # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
#             for i in range(2):
#                 for type in ["weight", "bias"]:
#                     old_key = f"{prefix}linear_{i+1}.{type}"
#                     new_key = f"{prefix}{3*i}.{type}"
#                     if old_key in state_dict:
#                         state_dict[new_key] = state_dict.pop(old_key)

#         super()._load_from_state_dict(
#             state_dict,
#             prefix,
#             local_metadata,
#             strict,
#             missing_keys,
#             unexpected_keys,
#             error_msgs,
#         )


# class EncoderBlock(nn.Module):
#     """Transformer encoder block."""

#     def __init__(
#         self,
#         num_heads: int,
#         hidden_dim: int,
#         mlp_dim: int,
#         dropout: float,
#         attention_dropout: float,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#     ):
#         super().__init__()
#         self.num_heads = num_heads

#         # Attention block
#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
#         self.dropout = nn.Dropout(dropout)

#         # MLP block
#         self.ln_2 = norm_layer(hidden_dim)
#         self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

#     def forward(self, input: torch.Tensor):
#         torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
#         x = self.ln_1(input)
#         x, _ = self.self_attention(x, x, x, need_weights=False)
#         x = self.dropout(x)
#         x = x + input

#         y = self.ln_2(x)
#         y = self.mlp(y)
#         return x + y


# class Encoder(nn.Module):
#     """Transformer Model Encoder for sequence to sequence translation."""

#     def __init__(
#         self,
#         seq_length: int,
#         num_layers: int,
#         num_heads: int,
#         hidden_dim: int,
#         mlp_dim: int,
#         dropout: float,
#         attention_dropout: float,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#     ):
#         super().__init__()
#         # Note that batch_size is on the first dim because
#         # we have batch_first=True in nn.MultiAttention() by default
#         self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
#         self.dropout = nn.Dropout(dropout)
#         layers: OrderedDict[str, nn.Module] = OrderedDict()
#         for i in range(num_layers):
#             layers[f"encoder_layer_{i}"] = EncoderBlock(
#                 num_heads,
#                 hidden_dim,
#                 mlp_dim,
#                 dropout,
#                 attention_dropout,
#                 norm_layer,
#             )
#         self.layers = nn.Sequential(layers)
#         self.ln = norm_layer(hidden_dim)

#     def forward(self, input: torch.Tensor):
#         torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
#         input = input + self.pos_embedding
#         return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(timm.models.vision_transformer.VisionTransformer,Backbone):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, out_feature="last_feat", **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.ps =  kwargs["patch_size"]
        self._out_feature_channels = {out_feature: kwargs["embed_dim"]}
        self._out_feature_strides = {out_feature: kwargs["patch_size"]}
        self._out_features = [out_feature]
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B,C,H,W = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = x[:,1:].view(B, H//self.ps, W//self.ps, -1)
        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x[:, 0]
        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs
    
    def forward(self, x):
        x = self.forward_features(x)
        # if self.head_dist is not None:
        #     x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
        #     if self.training and not torch.jit.is_scripting():
        #         # during inference, return the average of both classifier predictions
        #         return x, x_dist
        #     else:
        #         return (x + x_dist) / 2
        # else:
        #     x = self.head(x)
        return x



# class VisionTransformer(nn.Module):
#     """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

#     def __init__(
#         self,
#         image_size: int,
#         patch_size: int,
#         num_layers: int,
#         num_heads: int,
#         hidden_dim: int,
#         mlp_dim: int,
#         dropout: float = 0.0,
#         attention_dropout: float = 0.0,
#         num_classes: int = 1000,
#         representation_size: Optional[int] = None,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         conv_stem_configs: Optional[List[ConvStemConfig]] = None,
#     ):
#         super().__init__()
#         # _log_api_usage_once(self)
#         torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.hidden_dim = hidden_dim
#         self.mlp_dim = mlp_dim
#         self.attention_dropout = attention_dropout
#         self.dropout = dropout
#         self.num_classes = num_classes
#         self.representation_size = representation_size
#         self.norm_layer = norm_layer

#         if conv_stem_configs is not None:
#             # As per https://arxiv.org/abs/2106.14881
#             seq_proj = nn.Sequential()
#             prev_channels = 3
#             for i, conv_stem_layer_config in enumerate(conv_stem_configs):
#                 seq_proj.add_module(
#                     f"conv_bn_relu_{i}",
#                     Conv2dNormActivation(
#                         in_channels=prev_channels,
#                         out_channels=conv_stem_layer_config.out_channels,
#                         kernel_size=conv_stem_layer_config.kernel_size,
#                         stride=conv_stem_layer_config.stride,
#                         norm_layer=conv_stem_layer_config.norm_layer,
#                         activation_layer=conv_stem_layer_config.activation_layer,
#                     ),
#                 )
#                 prev_channels = conv_stem_layer_config.out_channels
#             seq_proj.add_module(
#                 "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
#             )
#             self.conv_proj: nn.Module = seq_proj
#         else:
#             self.conv_proj = nn.Conv2d(
#                 in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
#             )

#         seq_length = (image_size // patch_size) ** 2

#         # Add a class token
#         self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
#         seq_length += 1

#         self.encoder = Encoder(
#             seq_length,
#             num_layers,
#             num_heads,
#             hidden_dim,
#             mlp_dim,
#             dropout,
#             attention_dropout,
#             norm_layer,
#         )
#         self.seq_length = seq_length

#         heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
#         if representation_size is None:
#             heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
#         else:
#             heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
#             heads_layers["act"] = nn.Tanh()
#             heads_layers["head"] = nn.Linear(representation_size, num_classes)

#         self.heads = nn.Sequential(heads_layers)

#         if isinstance(self.conv_proj, nn.Conv2d):
#             # Init the patchify stem
#             fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
#             nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
#             if self.conv_proj.bias is not None:
#                 nn.init.zeros_(self.conv_proj.bias)
#         elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
#             # Init the last 1x1 conv of the conv stem
#             nn.init.normal_(
#                 self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
#             )
#             if self.conv_proj.conv_last.bias is not None:
#                 nn.init.zeros_(self.conv_proj.conv_last.bias)

#         if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
#             fan_in = self.heads.pre_logits.in_features
#             nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
#             nn.init.zeros_(self.heads.pre_logits.bias)

#         if isinstance(self.heads.head, nn.Linear):
#             nn.init.zeros_(self.heads.head.weight)
#             nn.init.zeros_(self.heads.head.bias)

#     def _process_input(self, x: torch.Tensor) -> torch.Tensor:
#         n, c, h, w = x.shape
#         p = self.patch_size
#         torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
#         torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
#         n_h = h // p
#         n_w = w // p

#         # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
#         x = self.conv_proj(x)
#         # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
#         x = x.reshape(n, self.hidden_dim, n_h * n_w)

#         # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
#         # The self attention layer expects inputs in the format (N, S, E)
#         # where S is the source sequence length, N is the batch size, E is the
#         # embedding dimension
#         x = x.permute(0, 2, 1)

#         return x

#     def forward(self, x: torch.Tensor):
#         # Reshape and permute the input tensor
#         x = self._process_input(x)
#         n = x.shape[0]

#         # Expand the class token to the full batch
#         batch_class_token = self.class_token.expand(n, -1, -1)
#         x = torch.cat([batch_class_token, x], dim=1)

#         x = self.encoder(x)

#         # Classifier "token" as used by standard language architectures
#         x = x[:, 0]

#         x = self.heads(x)

#         return x


# def _vision_transformer(
#     patch_size: int,
#     num_layers: int,
#     num_heads: int,
#     hidden_dim: int,
#     mlp_dim: int,
#     weights: None,
#     progress: bool,
#     **kwargs: Any,
# ) -> VisionTransformer:
#     # if weights is not None:
#     #     _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
#     #     assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
#     #     _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
#     image_size = kwargs.pop("image_size", 224)

#     model = VisionTransformer(
#         image_size=image_size,
#         patch_size=patch_size,
#         num_layers=num_layers,
#         num_heads=num_heads,
#         hidden_dim=hidden_dim,
#         mlp_dim=mlp_dim,
#         **kwargs,
#     )

#     if weights:
#         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

#     return model


# def vit_b_16(*, weights = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
#     """
#     Constructs a vit_b_16 architecture from
#     `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

#     Args:
#         weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
#             weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
#             below for more details and possible values. By default, no pre-trained weights are used.
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
#             for more details about this class.

#     .. autoclass:: torchvision.models.ViT_B_16_Weights
#         :members:
#     """
#     # weights = ViT_B_16_Weights.verify(weights)

#     return _vision_transformer(
#         patch_size=16,
#         num_layers=12,
#         num_heads=12,
#         hidden_dim=768,
#         mlp_dim=3072,
#         weights=weights,
#         progress=progress,
#         **kwargs,
#     )


# def vit_b_32(*, weights = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
#     """
#     Constructs a vit_b_32 architecture from
#     `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

#     Args:
#         weights (:class:`~torchvision.models.ViT_B_32_Weights`, optional): The pretrained
#             weights to use. See :class:`~torchvision.models.ViT_B_32_Weights`
#             below for more details and possible values. By default, no pre-trained weights are used.
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
#             for more details about this class.

#     .. autoclass:: torchvision.models.ViT_B_32_Weights
#         :members:
#     """
#     # weights = ViT_B_32_Weights.verify(weights)

#     return _vision_transformer(
#         patch_size=32,
#         num_layers=12,
#         num_heads=12,
#         hidden_dim=768,
#         mlp_dim=3072,
#         weights=weights,
#         progress=progress,
#         **kwargs,
#     )


# def vit_l_16(*, weights = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
#     """
#     Constructs a vit_l_16 architecture from
#     `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

#     Args:
#         weights (:class:`~torchvision.models.ViT_L_16_Weights`, optional): The pretrained
#             weights to use. See :class:`~torchvision.models.ViT_L_16_Weights`
#             below for more details and possible values. By default, no pre-trained weights are used.
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
#             for more details about this class.

#     .. autoclass:: torchvision.models.ViT_L_16_Weights
#         :members:
#     """
#     # weights = ViT_L_16_Weights.verify(weights)

#     return _vision_transformer(
#         patch_size=16,
#         num_layers=24,
#         num_heads=16,
#         hidden_dim=1024,
#         mlp_dim=4096,
#         weights=weights,
#         progress=progress,
#         **kwargs,
#     )


# def vit_l_32(*, weights = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
#     """
#     Constructs a vit_l_32 architecture from
#     `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

#     Args:
#         weights (:class:`~torchvision.models.ViT_L_32_Weights`, optional): The pretrained
#             weights to use. See :class:`~torchvision.models.ViT_L_32_Weights`
#             below for more details and possible values. By default, no pre-trained weights are used.
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
#             for more details about this class.

#     .. autoclass:: torchvision.models.ViT_L_32_Weights
#         :members:
#     """
#     # weights = ViT_L_32_Weights.verify(weights)

#     return _vision_transformer(
#         patch_size=32,
#         num_layers=24,
#         num_heads=16,
#         hidden_dim=1024,
#         mlp_dim=4096,
#         weights=weights,
#         progress=progress,
#         **kwargs,
#     )


# def vit_h_14(*, weights =  None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
#     """
#     Constructs a vit_h_14 architecture from
#     `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

#     Args:
#         weights (:class:`~torchvision.models.ViT_H_14_Weights`, optional): The pretrained
#             weights to use. See :class:`~torchvision.models.ViT_H_14_Weights`
#             below for more details and possible values. By default, no pre-trained weights are used.
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
#             for more details about this class.

#     .. autoclass:: torchvision.models.ViT_H_14_Weights
#         :members:
#     """
#     # weights = ViT_H_14_Weights.verify(weights)

#     return _vision_transformer(
#         patch_size=14,
#         num_layers=32,
#         num_heads=16,
#         hidden_dim=1280,
#         mlp_dim=5120,
#         weights=weights,
#         progress=progress,
#         **kwargs,
#     )


# def interpolate_embeddings(
#     image_size: int,
#     patch_size: int,
#     model_state: "OrderedDict[str, torch.Tensor]",
#     interpolation_mode: str = "bicubic",
#     reset_heads: bool = False,
# ) -> "OrderedDict[str, torch.Tensor]":
#     """This function helps interpolate positional embeddings during checkpoint loading,
#     especially when you want to apply a pre-trained model on images with different resolution.

#     Args:
#         image_size (int): Image size of the new model.
#         patch_size (int): Patch size of the new model.
#         model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
#         interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
#         reset_heads (bool): If true, not copying the state of heads. Default: False.

#     Returns:
#         OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
#     """
#     # Shape of pos_embedding is (1, seq_length, hidden_dim)
#     pos_embedding = model_state["encoder.pos_embedding"]
#     n, seq_length, hidden_dim = pos_embedding.shape
#     if n != 1:
#         raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

#     new_seq_length = (image_size // patch_size) ** 2 + 1

#     # Need to interpolate the weights for the position embedding.
#     # We do this by reshaping the positions embeddings to a 2d grid, performing
#     # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
#     if new_seq_length != seq_length:
#         # The class token embedding shouldn't be interpolated, so we split it up.
#         seq_length -= 1
#         new_seq_length -= 1
#         pos_embedding_token = pos_embedding[:, :1, :]
#         pos_embedding_img = pos_embedding[:, 1:, :]

#         # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
#         pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
#         seq_length_1d = int(math.sqrt(seq_length))
#         if seq_length_1d * seq_length_1d != seq_length:
#             raise ValueError(
#                 f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
#             )

#         # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
#         pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
#         new_seq_length_1d = image_size // patch_size

#         # Perform interpolation.
#         # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
#         new_pos_embedding_img = nn.functional.interpolate(
#             pos_embedding_img,
#             size=new_seq_length_1d,
#             mode=interpolation_mode,
#             align_corners=True,
#         )

#         # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
#         new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

#         # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
#         new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
#         new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

#         model_state["encoder.pos_embedding"] = new_pos_embedding

#         if reset_heads:
#             model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
#             for k, v in model_state.items():
#                 if not k.startswith("heads"):
#                     model_state_copy[k] = v
#             model_state = model_state_copy

#     return model_state

# ## TODO FIX BUGS
# class SimpleFeaturePyramid(nn.Module):
#     """
#     This module implements SimpleFeaturePyramid in :paper:`vitdet`.
#     It creates pyramid features built on top of the input feature map.
#     """
#     def __init__(
#         self,
#         net,
#         in_feature,
#         out_channels,
#         scale_factors,
#         top_block=None,
#         norm="LN",
#         square_pad=0,
#     ):
#         """
#         :param net (Backbone): module representing the subnetwork backbone.
#                 Must be a subclass of :class:`Backbone`.
#         :param in_feature (str): names of the input feature maps coming
#                 from the net.
#         :param out_channels (int): number of channels in the output feature maps.
#         :param scale_factors (list[float]): list of scaling factors to upsample or downsample
#                 the input features for creating pyramid features.
#         :param top_block (nn.Module or None): if provided, an extra operation will
#                 be performed on the output of the last (smallest resolution)
#                 pyramid output, and the result will extend the result list. The top_block
#                 further downsamples the feature map. It must have an attribute
#                 "num_levels", meaning the number of extra pyramid levels added by
#                 this block, and "in_feature", which is a string representing
#                 its input feature (e.g., p5).
#         :param norm (str): the normalization to use.
#         :param square_pad (int): If > 0, require input images to be padded to specific square size.
#         """
#         super(SimpleFeaturePyramid, self).__init__()
#         assert isinstance(net, nn.Module)
#         self.scale_factors = scale_factors
#         input_shapes = net.output_shape()
#         strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
#         # _assert_strides_are_log2_contiguous(strides)
#         dim = input_shapes[in_feature].channels
#         self.stages = []
#         use_bias = norm == ""
#         for idx, scale in enumerate(scale_factors):
#             out_dim = dim
#             if scale == 4.0:
#                 layers = [
#                     nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
#                     LayerNorm(dim // 2),
#                     nn.GELU(),
#                     nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
#                 ]
#                 out_dim = dim // 4
#             elif scale == 2.0:
#                 layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
#                 out_dim = dim // 2
#             elif scale == 1.0:
#                 layers = []
#             elif scale == 0.5:
#                 layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
#             layers.extend(
#                 [
#                     nn.Conv2d(
#                         out_dim,
#                         out_channels,
#                         kernel_size=1,
#                         bias=use_bias,
#                         # norm=get_norm(norm, out_channels),
#                     ),
#                     nn.Conv2d(
#                         out_channels,
#                         out_channels,
#                         kernel_size=3,
#                         padding=1,
#                         bias=use_bias,
#                         # norm=get_norm(norm, out_channels),
#                     ),
#                 ]
#             )
#             layers = nn.Sequential(*layers)
#             stage = int(math.log2(strides[idx]))
#             self.add_module(f"simfp_{stage}", layers)
#             self.stages.append(layers)
#         self.net = net
#         self.in_feature = in_feature
#         self.top_block = top_block
#         # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
#         self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
#         # top block output feature maps.
#         if self.top_block is not None:
#             for s in range(stage, stage + self.top_block.num_levels):
#                 self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
#         self._out_features = list(self._out_feature_strides.keys())
#         self._out_feature_channels = {k: out_channels for k in self._out_features}
#         self._size_divisibility = strides[-1]
#         self._square_pad = square_pad
#     @property
#     def padding_constraints(self):
#         return {
#             "size_divisiblity": self._size_divisibility,
#             "square_size": self._square_pad,
#         }
#     def forward(self, x):
#         """
#         :param x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
#         Returns:
#             dict[str->Tensor]:
#                 mapping from feature map name to pyramid feature map tensor
#                 in high to low resolution order. Returned feature names follow the FPN
#                 convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
#                 ["p2", "p3", ..., "p6"].
#         """
#         bottom_up_features = self.net(x)
#         features = bottom_up_features[self.in_feature]
#         results = []
#         for stage in self.stages:
#             results.append(stage(features))
#         if self.top_block is not None:
#             if self.top_block.in_feature in bottom_up_features:
#                 top_block_in_feature = bottom_up_features[self.top_block.in_feature]
#             else:
#                 top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
#             results.extend(self.top_block(top_block_in_feature))
#         assert len(self._out_features) == len(results)
#         return {f: res for f, res in zip(self._out_features, results)}

# class LayerNorm(nn.Module):
#     """
#     A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
#     variance normalization over the channel dimension for inputs that have shape
#     (batch_size, channels, height, width).
#     https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
#     """

#     def __init__(self, normalized_shape, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.normalized_shape = (normalized_shape,)

#     def forward(self, x):
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None] * x + self.bias[:, None, None]
#         return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed