import os
from numpy import dtype
import torch
import pickle

class RocketLeagueReplayDataset(torch.utils.data.Dataset):
    """Loads pickled replay data"""
    def __init__(self, root_dir = os.path.join("data", "training"), transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the bin uuid directories.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._root_dir = root_dir

        self._bins = os.listdir(root_dir)
        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self._bins)

    def __getitem__(self, idx):
        filename = os.path.join(self._root_dir, self._bins[idx])

        with open(filename, "rb") as f:
            contents = pickle.load(f)

        inputs = torch.stack(contents["inputs"])
        labels = torch.tensor(contents["labels"], dtype=torch.float32)
        labels = labels.view(*labels.shape, 1)

        if self._transform:
            inputs = self._transform(inputs)
        if self._target_transform:
            labels = self._target_transform(labels)

        return inputs, labels

if __name__ == "__main__":
    ds = RocketLeagueReplayDataset()
    inputs, labels = ds[0]
    print(f"inputs size: {inputs.size()}")
    print(f"labels size: {labels.size()}")
    print()
