import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel
from modules.utils import set_requires_grad

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        if self.visual_extractor == 'vit':
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.model.eval()
            set_requires_grad(self.model, False)
        else:
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        if self.visual_extractor != 'vit':
            patch_feats = self.model(images)
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            return patch_feats, avg_feats
        else:
            output = self.model(images)
            return output[0], output[1]

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, dropout=0.5, act='relu'):
        super(ResidualBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if act == 'relu' else nn.Identity()  # 使用ReLU作为激活函数

    def forward(self, x, f):
        residual = x + 0.001 * (f - x)
        return self.activation(self.dropout(residual))

class CombinedModel(nn.Module):
    def __init__(self, visual_extractor, feature_enhancer, in_dim=2048, dropout=0.5, act='relu'):
        super(CombinedModel, self).__init__()
        self.visual_extractor = visual_extractor
        self.feature_enhancer = feature_enhancer
        self.residual_block = ResidualBlock(in_dim=in_dim, dropout=dropout, act=act)

    def forward(self, images):
        patch_feats, avg_feats = self.visual_extractor(images)

        enhanced_feats = self.feature_enhancer(patch_feats)
        enhanced_feats = enhanced_feats[0]
        enhanced_feats = self.residual_block(patch_feats, enhanced_feats)
        return enhanced_feats, avg_feats









