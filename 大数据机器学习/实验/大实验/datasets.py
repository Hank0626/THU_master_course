from torch.utils.data import Dataset
from PIL import Image
import os


class PACS_all(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.domain = os.listdir(self.root_dir)
        self.label = os.listdir(os.path.join(self.root_dir, self.domain[0]))

        self.labels = []
        self.images = []
        for domain in self.domain:
            for label in self.label:
                for image in os.listdir(os.path.join(self.root_dir, domain, label)):
                    self.labels.append(label)
                    self.images.append(os.path.join(domain, label, image))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path)
        label = self.labels[index]

        if self.transform:
            sample = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label


class PACS_singledomain(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label = os.listdir(self.root_dir)

        self.labels = []
        self.images = []
        for label in self.label:
            for image in os.listdir(os.path.join(self.root_dir, label)):
                self.labels.append(label)
                self.images.append(os.path.join(label, image))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path)
        label = self.labels[index]

        if self.transform:
            sample = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label


class PACS_test(Dataset):  # 测试数据集
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)
        if ".ipynb_checkpoints" in self.images:
            pop_index = self.images.index(".ipynb_checkpoints")
            self.images.pop(pop_index)
        self.images = sorted(self.images, key=lambda x: int(x.split(".")[0]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path)

        if self.transform:
            sample = self.transform(img)
        return index, sample
