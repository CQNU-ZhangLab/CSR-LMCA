import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor
from modules.visual_extractor import CombinedModel
from modules.MambaMIL import MambaMIL

class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.layer = args.m_layer
        self.visual_extractor = VisualExtractor(args)
        # add MambaMIL
        self.feature_enhancer = MambaMIL(in_dim=2048, n_classes=10, dropout=0.5, act='relu', survival=False, layer=self.layer,
                                    rate=10)
        self.enhanced_feats = CombinedModel(self.visual_extractor, self.feature_enhancer, in_dim=2048, dropout=0.5, act='relu')
        
        self.linear_1 = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.ReLU()
        )
        self.linear_2 = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.ReLU()
        )

        self.encoder_decoder = BaseCMN(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif args.dataset_name == 'mimic_cxr':
            self.forward = self.forward_mimic_cxr
        elif args.dataset_name == 'mimic_cxr_2':
            self.forward = self.forward_mimic_cxr_dual

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):

        att_feats_0, fc_feats_0 = self.enhanced_feats(images[:, 0])
        att_feats_1, fc_feats_1 = self.enhanced_feats(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr_dual(self, images, targets=None, mode='train', update_opts={}):
        att_feats_0, fc_feats_0 = self.enhanced_feats(images[:, 0])
        att_feats_1, fc_feats_1 = self.enhanced_feats(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
