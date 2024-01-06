from datasets import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import cv2


class CannyEdgeDetection:
    def __init__(self, threshold1, threshold2, invert=True):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.invert = invert

    def __call__(self, img):
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, self.threshold1, self.threshold2)

        if self.invert:
            edges = cv2.bitwise_not(edges)

        edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        edges_3channel = Image.fromarray(edges_3channel)
        return edges_3channel


data_transforms = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        CannyEdgeDetection(150, 300),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

data_transforms_test = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

label_dict = {
    "dog": 0,
    "elephant": 1,
    "giraffe": 2,
    "guitar": 3,
    "horse": 4,
    "house": 5,
    "person": 6,
}
target_transform = lambda x: label_dict[x]


def get_dataloader_all(batch_size, num_workers):  # 所有训练域的dataloader
    PACS_dataset = PACS_all("PACS/train/", data_transforms, target_transform)
    PACS_dataloader = DataLoader(
        PACS_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    return PACS_dataloader


def get_dataloader_bydomain(batch_size, num_workers, domain):
    PACS_single_dataset = PACS_singledomain(
        f"PACS/train/{domain}", data_transforms, target_transform
    )

    PACS_single_dataloader = DataLoader(
        PACS_single_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    return PACS_single_dataloader


def get_dataloader_test(batch_size, num_workers):
    test_dataset = PACS_test("PACS/test", data_transforms_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return test_dataloader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
