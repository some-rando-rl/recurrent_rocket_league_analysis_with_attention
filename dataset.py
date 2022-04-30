import os
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

        input = torch.stack(contents["inputs"])
        label = torch.tensor(contents["labels"])

        if self._transform:
            input = self._transform(input)
        if self._target_transform:
            label = self._target_transform(label)

        return input, label

if __name__ == "__main__":
    ds = RocketLeagueReplayDataset()
    input, label = ds[0]
    print(f"input size: {input.size()}")
    print(f"label size: {label.size()}")
    print()
