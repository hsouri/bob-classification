import torch
import torch.nn as nn

import clip
from PIL import Image

#print(clip.available_models())
#['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, drop_path=None):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

        self.drop_path = drop_path

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.drop_path is not None:
            out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :]#.to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, drop_path_rate=0.0):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.drop_path_rate = drop_path_rate
        self.net_num_blocks = sum(layers)

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)


    def _make_layer(self, planes, blocks, stride=1, drop_out_rate=0.0):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for net_block_idx in range(1, blocks):
            block_dpr = self.drop_path_rate * net_block_idx / (self.net_num_blocks - 1)
            layers.append(Bottleneck(self._inplanes, planes, drop_path=DropPath(block_dpr) if block_dpr > 0. else None))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        #x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        #x = x.mean([-2, -1])

        return x


class ResNet_Clip(nn.Module):
    '''
    CLIP pretrained resnet models. The input of the model are tensors, which is preprocessed and tokenized.
    '''
    def __init__(self, embed_dim: int, image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int, num_classes=1000, drop_path_rate=0.0, **kwargs
                 ):
        super().__init__()

        vision_heads = vision_width * 32 // 64
        self.feature_extractor = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width,
            drop_path_rate=drop_path_rate
        )

        self.head = nn.Linear(embed_dim, num_classes, bias=True)
        #self.norm = nn.BatchNorm2d(embed_dim)
        self.init_weights()

    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

        trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)


    def forward_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x, text=None):
        x = self.forward_features(x)
        #x = self.norm(x)
        x = self.head(x)
        return x


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


def load_model(model_name):
    model, preprocess = clip.load(model_name, download_root='/fs/cml-projects/benchmarking_backbone/checkpoints/')
    state_dict = model.state_dict()

    counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
    vision_layers = tuple(counts)
    vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    image_resolution = output_width * 32

    return vision_layers, vision_width, image_resolution, model.visual.state_dict()


@register_model
def resnet50_clip(pretrained=False, image_res=224, **kwargs):
    vision_layers, vision_width, image_resolution, state_dict = load_model('RN50')

    if image_res is not None:
        image_resolution = image_res

    model = ResNet_Clip(image_resolution=image_resolution, vision_layers=vision_layers, vision_width=vision_width, embed_dim=1024, **kwargs)

    if pretrained:
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = 'feature_extractor.' + key
            new_state_dict[new_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    return model


@register_model
def resnet101_clip(pretrained=False, image_res=224, **kwargs):
    vision_layers, vision_width, image_resolution, state_dict = load_model('RN101')

    if image_res is not None:
        image_resolution = image_res

    model = ResNet_Clip(image_resolution=image_resolution, vision_layers=vision_layers, vision_width=vision_width, embed_dim=512, **kwargs)

    if pretrained:
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = 'feature_extractor.' + key
            new_state_dict[new_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    return model


@register_model
def resnet50_4_clip(pretrained=False, image_res=288, **kwargs):
    vision_layers, vision_width, image_resolution, state_dict = load_model('RN50x4')

    if image_res is not None:
        image_resolution = image_res

    model = ResNet_Clip(image_resolution=image_resolution, vision_layers=vision_layers, vision_width=vision_width, embed_dim=640, **kwargs)

    if pretrained:
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = 'feature_extractor.' + key
            new_state_dict[new_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    return model



@register_model
def resnet50_16_clip(pretrained=False, image_res=384, **kwargs):
    vision_layers, vision_width, image_resolution, state_dict = load_model('RN50x16')

    if image_res is not None:
        image_resolution = image_res

    model = ResNet_Clip(image_resolution=image_resolution, vision_layers=vision_layers, vision_width=vision_width, embed_dim=768, **kwargs)

    if pretrained:
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = 'feature_extractor.' + key
            new_state_dict[new_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    return model



@register_model
def resnet50_64_clip(pretrained=False, image_res=448, **kwargs):
    vision_layers, vision_width, image_resolution, state_dict = load_model('RN50x64')

    if image_res is not None:
        image_resolution = image_res

    model = ResNet_Clip(image_resolution=image_resolution, vision_layers=vision_layers, vision_width=vision_width, embed_dim=1024, **kwargs)

    if pretrained:
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = 'feature_extractor.' + key
            new_state_dict[new_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    return model



if __name__ == '__main__':
    mode = resnet50_clip(pretrained=True)
