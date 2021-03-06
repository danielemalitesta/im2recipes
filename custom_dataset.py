from torch.utils.data import Dataset
from PIL import ImageFile
from PIL import Image
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomDataset(Dataset):
    """
    This class represents a dataset. You can load it from the memory (by specifying its path)
    and use it for training/testing purposes.
    Attributes:
        root_dir (str): dataset path
        filenames (list): list of dataset filenames
        num_samples (int): number of samples inside the dataset
        transform: pre processing operations to perform on dataset
    """

    def __init__(self, root_dir, scale=2, reshape=False, transform=None):
        """
        Args:
            root_dir (str): dataset path
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = os.listdir(self.root_dir)
        self.num_samples = len(self.filenames)
        self.scale = scale
        self.reshape = reshape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = Image.open(self.root_dir + self.filenames[idx])
        height = sample.size[0]
        width = sample.size[1]

        try:
            sample.load()
        except:
            print(self.filenames[idx])

        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        if self.reshape:
            sample = sample.resize(size=(sample.size[0] // self.scale, sample.size[1] // self.scale),
                                   resample=Image.BICUBIC)

        if self.transform:
            sample = self.transform(sample)

        if self.reshape:
            return sample, height, width, self.filenames[idx]

        else:
            return sample, self.filenames[idx]
