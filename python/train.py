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

def quantize_symmetric_int8(tensor):
    array = tensor.detach().cpu().contiguous().numpy().astype("float32")

    max_abs = abs(array).max()

    if max_abs == 0:
        scale = 1.0
        q = array.astype("int8")

    else:
        scale = max_abs / 127.0
        q = (array / scale).round()
        q = q.clip(-127, 127)
        q = q.astype("int8")

    return q, scale

def export_tensor(tensor, path):
    array = tensor.detach().cpu().contiguous().numpy().astype("float32")
    array.tofile(path)
    print(f"wrote {path}: shape={tuple(array.shape)}, dtype={array.dtype}")


export_tensor(model.fc1.weight, "data/fc1_weight.bin")
export_tensor(model.fc1.bias,   "data/fc1_bias.bin")
export_tensor(model.fc2.weight, "data/fc2_weight.bin")
export_tensor(model.fc2.bias,   "data/fc2_bias.bin")

fc1_weight_q, fc1_weight_scale = quantize_symmetric_int8(model.fc1.weight)
fc2_weight_q, fc2_weight_scale = quantize_symmetric_int8(model.fc2.weight)

fc1_weight_q.tofile("data/fc1_weight_int8.bin")
fc2_weight_q.tofile("data/fc2_weight_int8.bin")

print("wrote data/fc1_weight_int8.bin:", fc1_weight_q.shape, fc1_weight_q.dtype)
print("wrote data/fc2_weight_int8.bin", fc2_weight_q.shape, fc2_weight_q.dtype)

print("fc1_weight_scale:", fc1_weight_scale)
print("fc2_weight_scale:", fc2_weight_scale)

print("fc1 original first 10:")
print(model.fc1.weight.detach().cpu().flatten()[:10])

print("fc1 int8 first 10:")
print(fc1_weight_q.flatten()[:10])

print("fc1 dequantized first 10:")
print((fc1_weight_q.astype("float32") * fc1_weight_scale).flatten()[:10])

with open("data/weight_scales.txt", "w") as f:
    f.write(f"{fc1_weight_scale}\n")
    f.write(f"{fc2_weight_scale}\n")

print("wrote data/weight_scales.txt")

image, label = test_data[0]

image_array = image.detach().cpu().contiguous().numpy().astype("float32")
image_array.tofile("data/test_image_0.bin")

with open("data/test_label_0.txt", "w") as f:
    f.write(str(label))

print("wrote data/test_image_0.bin")
print("wrote data/test_label_0.txt")


all_test_images = []
all_test_labels = []

for image, label in test_data:
    all_test_images.append(image)
    all_test_labels.append(label)

all_test_images = torch.stack(all_test_images) # [10000, 1, 28, 28]
all_test_labels = torch.tensor(all_test_labels, dtype=torch.int64)

print("all_test_images shape:", all_test_images.shape)
print("all_test_labels shape:", all_test_labels.shape)

all_test_images_array = all_test_images.detach().cpu().contiguous().numpy().astype("float32")
all_test_labels_array = all_test_labels.detach().cpu().contiguous().numpy().astype("int64")

all_test_images_array.tofile("data/test_images.bin")
all_test_labels_array.tofile("data/test_labels.bin")

print("wrote data/test_images.bin")
print("wrote data/test_labels.bin")