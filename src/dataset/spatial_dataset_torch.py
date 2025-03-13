from torch.utils.data import Dataset


class DistanceDataset(Dataset):
    def __init__(self, distance_matrix, y):
        self.distance_matrix = distance_matrix  # (n, n)
        self.y = y                              # (n, 1)
        self.n = distance_matrix.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.distance_matrix[index], self.y[index]
