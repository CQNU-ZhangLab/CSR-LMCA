"""
MambaMIL
"""
import torch
import torch.nn as nn
from mamba.mamba_ssm import SRMamba

import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False, layer=2, rate=10):
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.survival = survival

        for _ in range(layer):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(512),
                    SRMamba(
                        d_model=512,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    ),
                )
            )
        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.fc_to_2048 = nn.Linear(512, 2048)
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 256]

        for layer in self.layers:
            h_ = h
            h = layer[0](h)
            h = layer[1](h, rate=self.rate)
            h = h + h_


        h = self.norm(h)
        A = self.attention(h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(0)
        h = self.fc_to_2048(h)  # [B, 2048]
        return h, A, None, None, None

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)

        self.attention = self.attention.to(device)

        self.fc_to_2048 = self.fc_to_2048.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)


