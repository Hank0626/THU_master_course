import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import click
from utils import get_dataloader_all, get_dataloader_bydomain, setup_seed
from enhance import replace_bn_with_adabn, add_seblock_to_resnet

import warnings

warnings.filterwarnings("ignore")

MODEL_LIST = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    right_sample = 0
    total_sample = 0

    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = torch.argmax(output, dim=1)
        right_sample += torch.sum(pred == target)
        total_sample += target.shape[0]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    return train_loss / len(train_loader.sampler), right_sample / total_sample


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    right_sample = 0
    total_sample = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = torch.argmax(output, dim=1)
            right_sample += torch.sum(pred == target)
            total_sample += target.shape[0]
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)

    return val_loss / len(val_loader.sampler), right_sample / total_sample


@click.command()
@click.option("--epoch", default=10, help="Number of epochs.")
@click.option("--lr", default=0.001, help="Learning rate.")
@click.option("--weight_decay", default=0.0, help="Weight decay for optimizer.")
@click.option("--batch_size", default=64, help="Batch size.")
@click.option("--num_workers", default=64, help="Number of workers.")
@click.option("--model_name", default="resnet18", help="Model name.")
@click.option("--domain", default="all", help="Domain name.")
def main(epoch, lr, weight_decay, batch_size, num_workers, model_name, domain):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(0)

    assert domain in ["art_painting", "cartoon", "photo", "all"], "Domain name error!"

    assert model_name in MODEL_LIST, "Model name error!"

    if domain == "all":
        train_loader = get_dataloader_all(batch_size, num_workers)
    else:
        train_loader = get_dataloader_bydomain(batch_size, num_workers, domain)

    print(
        f"Train on {domain} domain, model: {model_name}, Train size: {len(train_loader.sampler)}"
    )

    n_class = 7

    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=False)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=False)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=False)

    replace_bn_with_adabn(model)

    add_seblock_to_resnet(model)

    model.fc = nn.Linear(model.fc.in_features, n_class)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0

    for ep in range(1, epoch + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(
            f"Epoch: {ep} \tTraining Loss: {train_loss:.6f}\tTraining Acc: {train_acc:.6f}"
        )

        torch.save(model.state_dict(), f"Checkpoints/{model_name}_{domain}.pth")
        print("Save model successfully!")


if __name__ == "__main__":
    main()
