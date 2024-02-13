from torch.utils.data import Dataset
from src.models.customdataset import CustomDataset


class TorchTrajectoryDataset(Dataset):
    def __init__(self, env, bb_model, path, k):
        self.ds = CustomDataset(env, bb_model, path, k)._dataset.sample(10000)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        i = self.ds.iloc[[idx]]
        x = i.values.squeeze()

        return x, x