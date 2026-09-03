# Dataset used for training the noise-robust classifier.

import numpy as np
import torch

class NoiseEmbeddingsDataset(torch.utils.data.Dataset):

    def __init__(self, latent_dim, num_embeddings, device, label):
        super(NoiseEmbeddingsDataset, self).__init__()

        self.device = device

        self.x = torch.randn(num_embeddings, latent_dim, device=device)
        self.y = torch.full(size=(num_embeddings,), fill_value=label).to(self.device)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]