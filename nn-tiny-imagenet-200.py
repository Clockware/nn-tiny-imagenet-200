import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import os
from shutil import copyfile

from torchvision import transforms, datasets
from collections import Counter


IS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if IS_CUDA else "cpu"


def with_progress_msg(msg, func):
    print("{0}... ".format(msg), end='', flush=True)
    result = func()
    print("Done.")
    return result


def ensure_dataset_loaded():
    if os.path.exists("tiny-imagenet-200"):
        return
    from tiny_img import download_tinyImg200
    data_path = '.'
    with_progress_msg("Downloading tiny-imagenet-200", lambda: download_tinyImg200(data_path))

    val_fixed_folder = "tiny-imagenet-200/val_fixed"
    if os.path.exists(val_fixed_folder):
        return
    os.mkdir(val_fixed_folder)

    with open("tiny-imagenet-200/val/val_annotations.txt") as f:
        for line in f.readlines():
            fields = line.split()

            file_name = fields[0]
            clazz = fields[1]

            class_folder = "tiny-imagenet-200/val_fixed/" + clazz
            if not os.path.exists(class_folder):
                os.mkdir(class_folder)

            original_image_path = "tiny-imagenet-200/val/images/" + file_name
            copied_image_path = class_folder + "/" + file_name

            copyfile(original_image_path, copied_image_path)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class FeatureCounter:
    def __init__(self, count):
        self.count = count


class ConvNormReLU(nn.Module):
    def __init__(self, f_counter, in_channels, out_channels=None, radius=1, stride=1):
        super(ConvNormReLU, self).__init__()

        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=radius * 2 + 1,
                               stride=stride,
                               padding=radius,
                               bias=False)
        f_counter.count *= out_channels
        f_counter.count //= in_channels
        f_counter.count //= stride * stride
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.relu:
            x = self.relu(x)

        return x


class Pool2d(nn.Module):
    def __init__(self, f_counter, clazz, radius=1, stride=1):
        super(Pool2d, self).__init__()

        f_counter.count //= stride * stride
        self.pool = clazz(kernel_size=1 + 2 * radius,
                          padding=radius,
                          stride=stride)

    def forward(self, x):
        return self.pool(x)


def get_model_with_opt(n_classes):
    seq_model = nn.Sequential()

    pixels = FeatureCounter(3 * 64 * 64)

    seq_model.add_module("Conv2D_Begin_Expand", ConvNormReLU(pixels, in_channels=3, out_channels=64, radius=1, stride=1))
    seq_model.add_module("MaxPooling2D", Pool2d(pixels, nn.MaxPool2d, radius=1, stride=2))

    seq_model.add_module("Conv2D_1_Expand", ConvNormReLU(pixels, in_channels=64, out_channels=128, radius=1, stride=1))
    seq_model.add_module("Conv2D_1_Reduce", ConvNormReLU(pixels, in_channels=128, out_channels=128, radius=1, stride=2))
    seq_model.add_module("Conv2D_2_Expand", ConvNormReLU(pixels, in_channels=128, out_channels=256, radius=1, stride=1))
    seq_model.add_module("Conv2D_2_Reduce", ConvNormReLU(pixels, in_channels=256, out_channels=256, radius=1, stride=2))

    seq_model.add_module("Conv2D_End_Expand", ConvNormReLU(pixels, in_channels=256, out_channels=512, radius=1, stride=1))
    seq_model.add_module("AvgPool2D", Pool2d(pixels, nn.AvgPool2d, radius=1, stride=2))
    seq_model.add_module("Flatten", Flatten())
    seq_model.add_module("OutputLayer", nn.Linear(pixels.count, n_classes))
    seq_model.add_module("OutputLayerSoftmax", nn.LogSoftmax(dim=1))

    seq_model.to(device=DEVICE)

    print("Weight shapes:", [w.shape for w in seq_model.parameters()])
    opt = torch.optim.Adam(seq_model.parameters())

    return seq_model, opt


def iterate_dataset(ds):
    return torch.utils.data.DataLoader(ds, batch_size=720, shuffle=True, num_workers=3)


def calculate_val_accuracy(val_dataset, model):
    val_total = 0
    val_success = 0
    for X_test_batch, y_test_batch in iterate_dataset(val_dataset):
        X_test_batch = X_test_batch.float().to(device=DEVICE)
        y_test_batch = y_test_batch.long().to(device=DEVICE)
        test_y_predicted = model(X_test_batch)
        val_total += len(y_test_batch)
        # noinspection PyUnresolvedReferences
        val_success += torch.sum(torch.argmax(test_y_predicted, dim=1) == y_test_batch).item()

    return val_success / val_total


def calculate_weights(dataset):
    class_count = Counter()

    for _, classes in iterate_dataset(dataset):
        for clazz in classes:
            class_count.update({clazz.item(): 1})

    total_classes = sum(class_count.values())

    weights = []
    for clazz in sorted(class_count.keys()):
        weights.append(class_count[clazz] / total_classes)

    return torch.Tensor(weights).float().to(device=DEVICE)


def learn(model, optimizer, train_dataset, val_dataset, patience=1):
    samples_for_progress_sign = len(train_dataset) // 50

    weights = with_progress_msg("Calculating weights", lambda: calculate_weights(train_dataset))

    val_accuracy_history = []
    best_epoch_number = 0
    while True:
        progress_signs_printed = 0
        learned_samples = 0
        for batch_idx, batch in enumerate(iterate_dataset(train_dataset)):
            X_batch, y_batch = batch

            X_batch = X_batch.float().to(device=DEVICE)
            y_batch = y_batch.long().to(device=DEVICE)

            y_predicted = model(X_batch)
            loss = nn.functional.nll_loss(y_predicted, y_batch, weight=weights)

            loss.backward()  # add new gradients
            optimizer.step()  # change weights
            optimizer.zero_grad()  # clear gradients

            learned_samples += len(X_batch)
            signs_expected = learned_samples / samples_for_progress_sign
            while progress_signs_printed < signs_expected:
                print("=", end='', flush=True)
                progress_signs_printed += 1

        val_accuracy = calculate_val_accuracy(val_dataset, model)
        print(" Epoch #{0} | {1:.2f}% accuracy".format(
            len(val_accuracy_history),
            100 * val_accuracy
        ))
        if not len(val_accuracy_history) or val_accuracy > np.max(val_accuracy_history):
            best_epoch_number = len(val_accuracy_history)
        if best_epoch_number + patience < len(val_accuracy_history):
            print("Time is against us!")
            break
        val_accuracy_history.append(val_accuracy)


def display_end_calculations(ds_name, ds, model):
    val_accuracy = calculate_val_accuracy(ds, model)
    print("{0}: {1:.2f}% accuracy".format(
        ds_name,
        100 * val_accuracy
    ))


def main():
    ensure_dataset_loaded()

    train_val_transform = transforms.Compose([
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder('tiny-imagenet-200/train', transform=train_val_transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [80000, 20000])

    model, optimizer = get_model_with_opt(len(dataset.classes))
    learn(model, optimizer, train_dataset, val_dataset)

    test_dataset = datasets.ImageFolder('tiny-imagenet-200/val_fixed', transform=test_transform)
    display_end_calculations("Train", train_dataset, model)
    display_end_calculations("Val", val_dataset, model)
    display_end_calculations("Test", test_dataset, model)


if __name__ == "__main__":
    main()

