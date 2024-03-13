from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2), "Datasets must be the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        item1 = self.dataset1[idx]
        item2 = self.dataset2[idx]
        return item1, item2
