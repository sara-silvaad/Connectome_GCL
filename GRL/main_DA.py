import torch.optim as optim
import torch
import os
from pathlib import Path
from data.dataset import Dataset
from data.dataset_loader import split_dataset_stratified, get_data_loaders
from training.train import train_and_evaluate
from tests.test import test_model
from utils.utils import check_path
from utils.plots import create_plots_and_save_results, save_average_results
from configs.config_DA import (
    model_mapping,
    LAMB,
    N,
    FC_PATHS,
    ROOT_PATH,
    MATRIX_FILE,
    PROB_DIS,
    DATASET_NAME,
    HIDDEN_DIMS,
    BATCH_SIZE,
)
from configs.config_DA import (
    EPOCHS,
    LR,
    MODEL,
    EARLY_STOPPING_PATIENCE,
    RUNS,
    RESULTS_PATH,
    RESIDUAL_CONNECTION,
    TRAIN_PROPORTION,
    LABELS_PATH,
)

MODEL = os.getenv("MODEL")

# Data
dataset = Dataset(
    root=ROOT_PATH,
    dataset_name=DATASET_NAME,
    matrix_file=MATRIX_FILE,
    fc_paths=FC_PATHS,
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
if MODEL == "EncoderDecoderSCAlternateDA":
    general_path = f"{RESULTS_PATH}/{MODEL}/{N}_{PROB_DIS}_{LAMB}"
else:
    general_path = f"{RESULTS_PATH}/{MODEL}/{N}_{PROB_DIS}"

general_path = check_path(general_path, config_path="configs/config_DA.py")
filename = (
    f"{MODEL}_{DATASET_NAME}_{BATCH_SIZE}_{EPOCHS}_{LR}_{EARLY_STOPPING_PATIENCE}"
)

ckpt_path = None

# Start training and testing RUNS amount of times
results = []
test_results = []
for run in range(RUNS):
    path_to_save = Path(general_path) / f"{run}"
    path_to_best_checkpoint = Path(path_to_save) / "ckpt" / f"best_{filename}.pt"

    print(f"Run {run}")
    model_class = model_mapping.get(MODEL)
    model = model_class(
        num_classes=dataset.num_classes,
        hidden_dims=HIDDEN_DIMS,
        model_name=MODEL,
        residual=RESIDUAL_CONNECTION,
        N=N,
        prob_of_diappearance=PROB_DIS,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:1")
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
        lamb=LAMB,
    )
    test_result = test_model(
        model,
        path_to_best_checkpoint,
        test_loader,
        criterion_recon,
        criterion_classif,
        lamb=LAMB,
    )

    create_plots_and_save_results(result, test_result, filename, path_to_save)

    results.append(result)
    test_results.append(test_result)

save_average_results(results, test_results, filename, general_path)
