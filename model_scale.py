import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import torch.nn as nn


class SqueezeChannels(nn.Module):
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class FCN(nn.Module):
    def __init__(self, num_classes, input_size=1):
        super(FCN, self).__init__()

        self.num_classes = num_classes
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.network = nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            nn.AdaptiveAvgPool1d(1),
            SqueezeChannels(),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x, scale_x=None, is_head=False):
        if is_head:
            embed_x = self.network(x)
            fea_concat = torch.cat((embed_x, scale_x, (embed_x - scale_x), (embed_x * scale_x)), dim=1)
            return self.projection_head(fea_concat)
        return self.network(x)


def conv_out_len(seq_len, ker_size, stride, dilation, stack):
    i = 0
    for _ in range(stack):
        seq_len = int((seq_len + (ker_size[i] - 1) - dilation * (ker_size[i] - 1) - 1) / stride + 1)
        i = i + 1
    return seq_len


class FCNDecoder(nn.Module):
    def __init__(self, num_classes, seq_len=None, input_size=None):
        super(FCNDecoder, self).__init__()

        self.num_classes = num_classes
        self.compressed_len = conv_out_len(seq_len=seq_len, ker_size=[3, 5, 7], stride=1, dilation=1, stack=3)

        self.conv_trans_block1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, output_padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv_trans_block2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=5, padding=2, output_padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.conv_trans_block3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=7, padding=3, output_padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.network = nn.Sequential(
            self.conv_trans_block1,
            self.conv_trans_block2,
            self.conv_trans_block3,

        )

        self.upsample = nn.Linear(1, self.compressed_len)
        self.conv1x1 = nn.Conv1d(128, input_size, 1)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        x = self.upsample(x)
        x = self.network(x)
        x = self.conv1x1(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dims, output_dims) -> None:
        super(Classifier, self).__init__()

        self.dense = nn.Linear(input_dims, output_dims)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.dense(x))
