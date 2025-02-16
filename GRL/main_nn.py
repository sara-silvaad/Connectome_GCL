import torch.optim as optim
import torch
import os
from pathlib import Path
from data.dataset import Dataset
from data.dataset_loader import get_data_loaders, split_dataset_stratified
from training.train import train_and_evaluate
from tests.test import test_model
from utils.plots import create_plots_and_save_results, save_average_results
from utils.utils import check_path
from configs.config_nn import (
    model_mapping,
    FC_PATHS,
    ROOT_PATH,
    MATRIX_FILE,
    DATASET_NAME,
    BATCH_SIZE,
    EPOCHS,
    LR,
    MODEL,
)
from configs.config_nn import (
    EARLY_STOPPING_PATIENCE,
    RUNS,
    RESULTS_PATH,
    LABELS_PATH,
    TRAIN_PROPORTION,
)

MODEL = os.getenv("MODEL")

# Data
dataset = Dataset(
    root=ROOT_PATH,
    dataset_name=DATASET_NAME,
    fc_paths=FC_PATHS,
    matrix_file=MATRIX_FILE,
    labels_path=LABELS_PATH,
)
train_dataset, val_dataset, test_dataset = split_dataset_stratified(dataset)

# Keep only TRAIN_PROPORTION of the train_dataset
train_size = int(TRAIN_PROPORTION * len(train_dataset))
train_dataset, _ = torch.utils.data.random_split(
    train_dataset, [train_size, len(train_dataset) - train_size]
)
print(
    f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
)
train_loader, val_loader, test_loader = get_data_loaders(
    train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE
)

# Solve path saving
general_path = f"{RESULTS_PATH}/{MODEL}/"
general_path = check_path(general_path, config_path="configs/config_nn.py")
filename = f"{MODEL}_{DATASET_NAME}_{BATCH_SIZE}_{EPOCHS}_{LR}"

ckpt_path = None

results = []
test_results = []

# Start training and testing RUNS amount of times
for run in range(RUNS):
    path_to_save = Path(general_path) / f"{run}"
    path_to_best_checkpoint = Path(path_to_save) / "ckpt" / f"best_{filename}.pt"

    print(f"Run {run}")

    model_class = model_mapping.get(MODEL)
    model = model_class()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print("Model and tensors moved to GPU.")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU.")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_recon = torch.nn.MSELoss()
    criterion_classif = torch.nn.CrossEntropyLoss(reduction="mean")

    result = train_and_evaluate(
        model,
        train_loader,
        val_loader,
        EPOCHS,
        optimizer,
        criterion_recon,
        criterion_classif,
        EARLY_STOPPING_PATIENCE,
        path_to_save,
        filename,
        ckpt_path=ckpt_path,
    )
    test_result = test_model(
        model, path_to_best_checkpoint, test_loader, criterion_recon, criterion_classif
    )

    create_plots_and_save_results(result, test_result, filename, path_to_save)

    results.append(result)
    test_results.append(test_result)

save_average_results(results, test_results, filename, general_path)
