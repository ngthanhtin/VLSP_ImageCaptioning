""" Transformer in Transformer (TNT) in PyTorch
A PyTorch implement of TNT as described in
'Transformer in Transformer' - https://arxiv.org/abs/2103.00112
The official mindspore code is released and available at
https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/TNT
"""
import torch

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.tnt import (
    TNT,
    default_cfgs,
)


class TNTEx(TNT):
    def __init__(self, sumup_multiscales=False, **kwargs):
        super().__init__(**kwargs)
        self.sumup_multiscales = sumup_multiscales

    def forward_features(self, x):
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)
        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        patch_embeds = []
        for blk in self.blocks:
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed)
            patch_embeds.append(patch_embed)
        patch_embed = self.norm(patch_embed)
        return patch_embed


@register_model
def tnt_s_patch16_224_ex(pretrained=False, **kwargs):
    model = TNTEx(patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
        qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def tnt_b_patch16_224_ex(pretrained=False, **kwargs):
    model = TNTEx(patch_size=16, embed_dim=640, in_dim=40, depth=12, num_heads=10, in_num_head=4,
        qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_b_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model