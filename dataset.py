import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Actual mean and standard deviation values used to train resnet on CIFAR10
# Batch size can be adjusted on stronger GPUs
mean = [0.4914, 0.4822, 0.4465] 
std = [0.2471, 0.2435, 0.2616]
batch_size = 128
num_workers = 2 

# For augmentation, 4 pixels are padded on each side and a (32*32) crop is taken
train_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root='data',
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,       
    num_workers=num_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,      
    num_workers=num_workers
)