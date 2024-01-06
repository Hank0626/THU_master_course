import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import click
from utils import get_dataloader_test
from enhance import replace_bn_with_adabn, add_seblock_to_resnet

import warnings

warnings.filterwarnings("ignore")

label_reverse_list = [
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
]


def get_model(model_name, n_class):
    model_name = model_name.split("_")[0]
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
    else:
        raise ValueError(f"Unknown model {model_name}")

    model.fc = nn.Linear(model.fc.in_features, n_class)
    replace_bn_with_adabn(model)
    add_seblock_to_resnet(model)
    return model


def softmax_outputs(model, test_loader, device):
    softmax = nn.Softmax(dim=1)
    all_outputs = []

    model.eval()
    with torch.no_grad():
        for _, data in test_loader:
            data = data.to(device)
            outputs = model(data)
            softmax_out = softmax(outputs).cpu()
            all_outputs.append(softmax_out)

    return torch.cat(all_outputs, dim=0)


@click.command()
@click.option("--batch_size", default=32, help="Batch size for the dataloader.")
@click.option("--num_workers", default=32, help="Number of workers for dataloader.")
@click.option(
    "--model_names", multiple=True, help="List of model names for prediction."
)
def main(batch_size, num_workers, model_names):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = get_dataloader_test(batch_size=batch_size, num_workers=num_workers)
    n_class = 7

    model_outputs = []
    for model_name in model_names:
        model = get_model(model_name, n_class)
        state_dict = torch.load(f"Checkpoints/{model_name}.pth")
        model.load_state_dict(state_dict)
        model.to(device)

        outputs = softmax_outputs(model, test_loader, device)
        model_outputs.append(outputs)

    average_output = torch.mean(torch.stack(model_outputs), dim=0)
    pred_labels = torch.argmax(average_output, dim=1)
    pred_labels = [label_reverse_list[i] for i in pred_labels]

    IDs = list(range(len(pred_labels)))
    results = pd.DataFrame({"ID": IDs, "label": pred_labels})
    results.to_csv("result.csv", index=False)

    print("Test finished! Predict file saved in result.csv")


if __name__ == "__main__":
    main()
