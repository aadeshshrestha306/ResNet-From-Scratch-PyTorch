import torch
from resnet import resnet34
from dataset import train_loader, test_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    model = resnet34.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)

    num_epochs = 5
    accuracy = []
    losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0.0
        total_size = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_size += labels.size(0)

        train_accuracy = 100*correct_preds/total_size
        accuracy.append(train_accuracy)
        train_loss = running_loss/ len(train_loader.dataset)
        losses.append(train_loss)

        print(f"Epoch={epoch+1}, Loss={loss:.4f}, Accuracy={train_accuracy:.2f}%")