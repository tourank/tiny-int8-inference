# python/export.py

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import TinyMLP


def load_test_data():
    return datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )


def export_tensor(tensor, path):
    array = tensor.detach().cpu().contiguous().numpy().astype("float32")
    array.tofile(path)
    print(f"wrote {path}: shape={tuple(array.shape)}, dtype={array.dtype}")


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


def export_fp32_weights(model):
    export_tensor(model.fc1.weight, "data/fc1_weight.bin")
    export_tensor(model.fc1.bias, "data/fc1_bias.bin")
    export_tensor(model.fc2.weight, "data/fc2_weight.bin")
    export_tensor(model.fc2.bias, "data/fc2_bias.bin")


def export_int8_weights(model):
    fc1_weight_q, fc1_weight_scale = quantize_symmetric_int8(model.fc1.weight)
    fc2_weight_q, fc2_weight_scale = quantize_symmetric_int8(model.fc2.weight)

    fc1_weight_q.tofile("data/fc1_weight_int8.bin")
    fc2_weight_q.tofile("data/fc2_weight_int8.bin")

    print("wrote data/fc1_weight_int8.bin:", fc1_weight_q.shape, fc1_weight_q.dtype)
    print("wrote data/fc2_weight_int8.bin:", fc2_weight_q.shape, fc2_weight_q.dtype)

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


def export_single_test_image(test_data):
    image, label = test_data[0]

    image_array = image.detach().cpu().contiguous().numpy().astype("float32")
    image_array.tofile("data/test_image_0.bin")

    with open("data/test_label_0.txt", "w") as f:
        f.write(str(label))

    print("wrote data/test_image_0.bin")
    print("wrote data/test_label_0.txt")


def export_all_test_data(test_data):
    all_test_images = []
    all_test_labels = []

    for image, label in test_data:
        all_test_images.append(image)
        all_test_labels.append(label)

    all_test_images = torch.stack(all_test_images)
    all_test_labels = torch.tensor(all_test_labels, dtype=torch.int64)

    print("all_test_images shape:", all_test_images.shape)
    print("all_test_labels shape:", all_test_labels.shape)

    all_test_images_array = all_test_images.detach().cpu().contiguous().numpy().astype("float32")
    all_test_labels_array = all_test_labels.detach().cpu().contiguous().numpy().astype("int64")

    all_test_images_array.tofile("data/test_images.bin")
    all_test_labels_array.tofile("data/test_labels.bin")

    print("wrote data/test_images.bin")
    print("wrote data/test_labels.bin")


def main():
    model = TinyMLP()
    model.load_state_dict(torch.load("data/model.pt"))
    model.eval()

    test_data = load_test_data()

    export_fp32_weights(model)
    export_int8_weights(model)
    export_single_test_image(test_data)
    export_all_test_data(test_data)


if __name__ == "__main__":
    main()