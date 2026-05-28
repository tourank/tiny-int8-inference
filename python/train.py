import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

torch.manual_seed(0)

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

batch_size = 64
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class TinyMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = TinyMLP()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(model, dataloader, loss_fn, optimizer):
    model.train()

    total_loss = 0.0

    for images, labels in dataloader:
        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

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

num_epochs = 3

for epoch in range(num_epochs):
    avg_loss = train(model, train_loader, loss_fn, optimizer)
    accuracy = evaluate(model, test_loader)

    print(f"epoch {epoch + 1}: loss={avg_loss:.4f}, test_accuracy={accuracy:.4f}")


image, label = test_data[0]

with torch.no_grad():
    logits = model(image.unsqueeze(0))
    prediction = logits.argmax(dim=1).item()

print("fc1.weight", model.fc1.weight.shape)
print("fc1.bias", model.fc1.bias.shape)
print("fc2.weight", model.fc2.weight.shape)
print("fc2.bias", model.fc2.bias.shape)


print("image shape:", image.shape)
print("batched image shape:", image.unsqueeze(0).shape)
print("logits shape:", logits.shape)
print("prediction:", prediction)
print("label:", label)

def export_tensor(tensor, path):
    array = tensor.detach().cpu().contiguous().numpy().astype("float32")
    array.tofile(path)
    print(f"wrote {path}: shape={tuple(array.shape)}, dtype={array.dtype}")


print("fc1.weight first 10:")
print(model.fc1.weight.detach().cpu().flatten()[:10])

export_tensor(model.fc1.weight, "data/fc1_weight.bin")
export_tensor(model.fc1.bias,   "data/fc1_bias.bin")
export_tensor(model.fc2.weight, "data/fc2_weight.bin")
export_tensor(model.fc2.bias,   "data/fc2_bias.bin")

image, label = test_data[0]

image_array = image.detach().cpu().contiguous().numpy().astype("float32")
image_array.tofile("data/test_image_0.bin")

with open("data/test_label_0.txt", "w") as f:
    f.write(str(label))

print("wrote data/test_image_0.bin")
print("wrote data/test_label_0.txt")

with torch.no_grad():
    x = model.flatten(image.unsqueeze(0))
    h_pre_relu = model.fc1(x)

print("fc1 pre-ReLU first 10:")
print(h_pre_relu.flatten()[:10])
