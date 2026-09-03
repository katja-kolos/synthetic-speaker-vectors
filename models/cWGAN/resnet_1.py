# v0: copy on 18.09.2025 from https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/blob/main/anonymization/modules/sttts/speaker_embeddings/anonymization/utils/train_gan_model.py


import numpy as np
import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import nn


class ResNet_G(nn.Module):

    def __init__(self, 
        data_dim, z_dim, size, 
        label_embedding_dim, class_size, 
        nfilter=64, nfilter_max=512, bn=True, res_ratio=0.1, dropout_rate = 0, **kwargs):

        super().__init__()
        self.input_dim = z_dim
        self.output_dim = z_dim
        self.dropout_rate = dropout_rate

        self.label_embedding = nn.Embedding(class_size, label_embedding_dim) # classes: [0, 1] -> 1-hot vectors or other vectors

        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.bn = bn
        self.z_dim = z_dim

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** (nlayers + 1))

        # self.fc = nn.Linear(z_dim, self.nf0 * s0 * s0)
        self.fc = nn.Linear(z_dim + label_embedding_dim, self.nf0 * s0 * s0)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0 * s0 * s0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2 ** (i + 1), nf_max)
            nf1 = min(nf * 2 ** i, nf_max)
            blocks += [
                ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
                nn.Upsample(scale_factor=2)
                ]

        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [
            ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
            ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio)
            ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

        self.fc_out = nn.Linear(3 * size * size, data_dim)

    def forward(self, z, labels, return_intermediate=False):
        # labels may in fact be textual or otherwise complicated -- we embed them
        # in our case labels are just 0 and 1 for the 2 gender values
        # we still embed them to have one single process from the start
        # we'll have a one-hot vector of size 2
        # labels come in as a list: [0, 1, 0...] of the length of the batch size
        label_vector = self.label_embedding(labels) # (B, y_dim)
        # they are then transformed into an embedding of (batch_size, label_embedding_dim)
        print(f"Shape of embedded labels: {label_vector.shape}")
        # we now concatenate those label embeddings with the input noise embedding (latent_vector)
        # print(f"Shape of noise before concatenation with label embedding: {z.shape}")
        #concatenate latent vector (input) and label
        zc = torch.cat([z, label_vector], dim=1) # (B, z_dim + y_dim)
        # print(f"Shape of noise after concatenation with label embedding: {zc.shape}")
        # this concatenated input is passed into the resnet to generate plausible data
        batch_size = zc.size(0)
        # print(f"Batch size: {batch_size}")
        out = self.fc(zc)
        if self.bn:
            out = self.bn1d(out)
        out = self.relu(out)
        # print(f"Shape of fully-connected output: {out.shape}")
        if return_intermediate:
            l_1 = out.detach().clone()
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        # print(f"Shape of fully-connected output transformed: {out.shape}")
        out = self.resnet(out)
        # print(f"Shape after resnet: {out.shape}")
        out = self.conv_img(out)
        out = self.relu(out)
        # print(f"Shape after conv_img: {out.shape}")
        out.flatten(1)
        # print(f"flattenned: {out.shape}")
        out = self.fc_out(out.flatten(1))
        # print(f"fc_out: {out.shape}")

        if return_intermediate:
            return out, l_1
        return out

    def sample_latent(self, labels, z_size):
    # def sample_latent(self, n_samples, z_size):
        # return torch.randn((n_samples, z_size))
        # v1: sample for labels instead of random noise
        # TODO!!
        return torch.randn((len(labels), z_size))


class ResNet_D(nn.Module):
    #class_size, embedding_dim, batch_size, input_size
    def __init__(self, 
                 data_dim, size, 
                 label_embedding_dim, class_size, 
                 nfilter=64, nfilter_max=512, res_ratio=0.1):
        super().__init__()
        # class_size: number of classes (2 for gender)
        self.label_embedding = nn.Embedding(class_size, label_embedding_dim) # classes: [0, 1, 2] -> 1-hot vectors or other vectors

        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.size = size

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)
        ]

        # self.fc_input = nn.Linear(data_dim, 3 * size * size)
        self.fc_input = nn.Linear(data_dim + label_embedding_dim, 3 * size * size)
        # 130 x 3* custom configuration of layer's width (?) ^ 2 

        for i in range(1, nlayers + 1):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)

        self.fc = nn.Linear(self.nf0 * s0 * s0, 1)

    def forward(self, speaker_vectors, labels):
        print("ResNet_D forward:")
        #embed label (one-hot?)        
        label_vector = self.label_embedding(labels) # (B, y_dim)
        # print(f"Shape of label_embedding: {label_vector.shape}")
        # maybe we need to flatten the "image" first, like below
        # x = x.view(x.size(0), -1)
        print(f"Shape of input speaker_vectors: {speaker_vectors.shape}")
        speaker_vectors = speaker_vectors.view(speaker_vectors.size(0), -1) # because of how I construct the dataset
        print(f"Shape of speaker_vectors after view: {speaker_vectors.shape}")
        batch_size = speaker_vectors.size(0)
        print(f"batch_size: {batch_size}")
        xc = torch.cat([speaker_vectors, label_vector], dim=1)
        print(f"label vector: {label_vector.shape}")
        print(f"concatenated xc: {xc.shape}")

        out = self.fc_input(xc)
        out = self.relu(out).view(batch_size, 3, self.size, self.size)

        out = self.relu((self.conv_img(out)))
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)

        return out


class ResNetBlock(nn.Module):

    def __init__(self, fin, fout, fhidden=None, bn=True, res_ratio=0.1):
        super().__init__()
        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_0 = nn.BatchNorm2d(self.fhidden)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_1 = nn.BatchNorm2d(self.fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn2d_s = nn.BatchNorm2d(self.fout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.relu(x_s + self.res_ratio * dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s
