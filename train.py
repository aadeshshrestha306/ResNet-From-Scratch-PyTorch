import torch
from resnet import resnet34
from dataset import train_loader, test_loader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = resnet34().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


