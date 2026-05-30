import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import TinyMLP

torch.manual_seed(0)

def load_data(batch_size):
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return training_data, test_data, train_loader, test_loader

def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()

    total_loss = 0.0

    for images, labels in dataloader:
        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    torch.manual_seed(0)

    batch_size = 64
    num_epochs = 3

    _, _, train_loader, test_loader = load_data(batch_size)

    model = TinyMLP()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        accuracy = evaluate(model, test_loader)

        print(f"epoch {epoch + 1}: loss={avg_loss:.4f}, test_accuracy={accuracy:.4f}")

    torch.save(model.state_dict(), "data/model.pt")
    print("wrote data/model.pt")

if __name__ == "__main__":
    main()