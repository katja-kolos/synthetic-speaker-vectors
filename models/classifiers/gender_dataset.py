import torch
from torch.utils.data import Dataset, DataLoader


# Dataset to learn a classifier of gender from speaker embeddings (not audio)
class GSTGenderDataset(Dataset):
    def __init__(self, embeddings, labels, device='cpu'):
        """
        embeddings: torch.tensor of shape (N, D)
        labels: torch.tensor of shape (N,), with values 0 (male) or 1 (female)
        """
        # convert just in case it came as np.array
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
