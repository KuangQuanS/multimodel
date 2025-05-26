import numpy as np
import torch
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    def __init__(self, npz_path, modalities, indices=None):
        data = np.load(npz_path)
        self.Xs = [data[mod] for mod in modalities]
        self.y = data['y']
        if indices is not None:
            self.Xs = [x[indices] for x in self.Xs]
            self.y = self.y[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [torch.FloatTensor(x[idx]) for x in self.Xs], torch.tensor(self.y[idx], dtype=torch.long)