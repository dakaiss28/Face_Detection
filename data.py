import torch
import torchvision
import torchvision.transforms as transforms

import os.path

imagenet_preprocessing = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

image_transform = transforms.Compose([transforms.ToTensor(), imagenet_preprocessing])

dataset_dir = os.path.join(os.path.expanduser("~"), "Datasets", "WiderFACE")


# training set
train_dataset = torchvision.datasets.WIDERFace(
    root=dataset_dir, split="train", transform=image_transform, download=True
)


# Ltest set
test_dataset = torchvision.datasets.WIDERFace(
    root=dataset_dir, transform=image_transform, split="test"
)

num_threads = 2
batch_size = 128

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_threads,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_threads,
)
