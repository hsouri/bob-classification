from models.midas.model_loader import default_models, load_model
from models.midas.dpt_depth import DPT, DPTDepthModel

from functools import partial

import torch
import torch.nn as nn

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

model_urls = {
     "swin2_tiny_256_midas": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt",
     "dpt_swin2_base_384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt",
    }

import pdb

@register_model
def swin2_tiny_256_midas(pretrained=False, **kwargs):
    if pretrained:
        path = model_urls['swin2_tiny_256_midas']
    else:
        path = None

    midas_model = DPTDepthModel(
        path=path,
        backbone="swin2t16_256",
        non_negative=True,
    )
    num_features = 768
    model = midas_model.pretrained.model
    return model

@register_model
def swin2_base_384_midas(pretrained=False, **kwargs):
    if pretrained:
        path = model_urls['dpt_swin2_base_384']
    else:
        path = None

    midas_model = DPTDepthModel(
        path=path,
        backbone="swin2b24_384",
        non_negative=True,
    )
    num_features = 768
    model = midas_model.pretrained.model
    return model

