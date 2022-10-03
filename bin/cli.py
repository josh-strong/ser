import json
from datetime import datetime
from pathlib import Path

import git
import torch
import typer

from ser.constants import RESULTS_DIR
from ser.data import test_dataloader, train_dataloader, val_dataloader
from ser.evaluate import evaluate
from ser.params import Params, save_params
from ser.train import train as run_train
from ser.transforms import normalize, transforms

main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        5, "-e", "--epochs", help="Number of epochs to run for."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for dataloader."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning-rate", help="Learning rate for the model."
    ),
):
    """Run the training algorithm."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # wraps the passed in parameters
    params = Params(name, epochs, batch_size, learning_rate, sha)

    # setup device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup run
    fmt = "%Y-%m-%dT%H-%M"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = RESULTS_DIR / name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # Save parameters for the run
    save_params(run_path, params)

    # Train!
    run_train(
        run_path,
        params,
        train_dataloader(params.batch_size, transforms(normalize)),
        val_dataloader(params.batch_size, transforms(normalize)),
        device,
    )


@main.command()
def infer(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to use for inference."
    ),
    run: str = typer.Option(
        ..., "-r", "--run", help="Name of run to use for inference."
    ),
    label: int = typer.Option(
        ..., "-l", "--label", help="Label to inference."
    )

):
    run_path = Path("./results") / name / run

    # load the parameters from the run_path
    with open(run_path / "params.json") as f:
        params = json.load(f)

    # select image to run inference for
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model = torch.load(run_path / "model.pt")

    # run inference
    evaluate(model, images)

    # Print summary of experiment
    print(f"Experiment name: {name} \n Name of run: {run} \n")
    print("Model parameters:", params)
