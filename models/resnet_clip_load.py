import torch
import torch.nn as nn

import clip
from PIL import Image

#print(clip.available_models())
#['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

from timm.models.registry import register_model


def freeze_pretrained_layers(model):
    '''Freeze all layers except the last layer(fc or classifier)'''
    for param in model.parameters():
        param.requires_grad = False

    #nn.init.xavier_normal_(model.head.weight)
    #nn.init.zeros_(model.head.bias)

def del_transformer(model):
    del model.transformer

class ResNet_Clip(nn.Module):
    '''
    CLIP pretrained resnet models. The input of the model are tensors, which is preprocessed and tokenized.
    '''
    def __init__(self, model_name, num_classes, embed_dim, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model, self.preprocess = clip.load(self.model_name, download_root='/fs/cml-projects/benchmarking_backbone/checkpoints/')
        self.head = nn.Linear(embed_dim, num_classes, bias=True)

    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    def forward_features(self, x):
        return self.model.encode_image(x).float()

    def forward_text_features(self, x):
        return self.model.encode_text(x).float()

    #def forward_head(self, x, pre_logits: bool = False):
    #    x = self.global_pool(x)
    #    if self.drop_rate:
    #        x = F.dropout(x, p=float(self.drop_rate), training=self.training)
    #    return x if pre_logits else self.fc(x)

    def forward(self, x, text=None):
        x = self.forward_features(x)
        if text is not None:
            text_features = self.forward_text_features(text)
            return x, text_features
        x = self.head(x)
        return x


@register_model
def resnet50_clip_load(pretrained=False, num_classes=1000, **kwargs):
    model = ResNet_Clip('RN50', embed_dim=1024, num_classes=num_classes, **kwargs)
    #freeze_pretrained_layers(model.model.transformer)
    del_transformer(model.model)
    if not pretrained:
        model.init_weights()
    return model


@register_model
def resnet101_clip_load(pretrained=False, num_classes=1000, **kwargs):
    model = ResNet_Clip('RN101', embed_dim=512, num_classes=num_classes, **kwargs)
    freeze_pretrained_layers(model.model.transformer)
    if not pretrained:
        model.init_weights()
    return model


@register_model
def resnet50_4_clip_load(pretrained=False, **kwargs):
    model = ResNet_Clip('RN50x4', **kwargs)
    freeze_pretrained_layers(model.model.transformer)
    if not pretrained:
        model.init_weights()
    return model


@register_model
def resnet50_16_clip_load(pretrained=False, **kwargs):
    model = ResNet_Clip('RN50x16', **kwargs)
    freeze_pretrained_layers(model.model.transformer)
    if not pretrained:
        model.init_weights()
    return model


@register_model
def resnet50_64_clip_load(pretrained=False, **kwargs):
    model = ResNet_Clip('RN50x64', **kwargs)
    freeze_pretrained_layers(model.model.transformer)
    if not pretrained:
        model.init_weights()
    return model


