import torch.optim as optim
import os
from data.dataset import Dataset
from data.dataset_loader import get_data_loaders, split_dataset_stratified
from training.train import train_and_evaluate_pre
from tests.test import test_pre
from utils.plots import (
    create_plots_and_save_results,
    save_average_results,
    solve_paths_pre,
)
from config import (
    ROOT_PATH,
    MATRIX_FILE,
    DATASET_NAME,
    HIDDEN_DIMS,
    LABELS_PATH,
    BATCH_SIZE_pre,
    EPOCHS_pre,
    LR_pre,
    EARLY_STOPPING_PATIENCE_pre,
    RUNS_pre,
)
from config import (
    LAMB,
    FC_PATHS,
    model_mapping,
    DATA_AUG,
    SAVED_MODELS_PATH_pre,
    RESULTS_PATH_pre,
    TAU,
    TRAIN_PROPORTION,
)
import torch

DATA_AUG = os.getenv("DATA_AUG")

# Data
dataset = Dataset(
    root=ROOT_PATH,
    dataset_name=DATASET_NAME,
    matrix_file=MATRIX_FILE,
    fc_paths=FC_PATHS,
    labels_path=LABELS_PATH,
)
train_dataset, val_dataset, test_dataset = split_dataset_stratified(dataset)

# keep only TRAIN_PROPORTION of the train_dataset
train_size = int(TRAIN_PROPORTION * len(train_dataset))
train_dataset, _ = torch.utils.data.random_split(
    train_dataset, [train_size, len(train_dataset) - train_size]
)
print(
    f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
)

train_loader, val_loader, test_loader = get_data_loaders(
    train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE_pre
)

# Solve path saving
PRE_MODEL_PATH, PRE_RESULTS_PATH = solve_paths_pre(
    SAVED_MODELS_PATH_pre, RESULTS_PATH_pre, DATA_AUG, TAU
)
pre_model_name = f"{DATA_AUG}_{EPOCHS_pre}_{LR_pre}"

results = []
test_results = []

# Start training and testing RUNS amount of times
for run in range(RUNS_pre):
    print(f"Run {run}")

    model_class, criterion_recon = model_mapping.get(DATA_AUG)
    model = model_class(hidden_dims=HIDDEN_DIMS, num_classes=dataset.num_classes)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print("Model and tensors moved to GPU.")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU.")

    optimizer = optim.Adam(model.parameters(), lr=LR_pre)

    result, best_model = train_and_evaluate_pre(
        model,
        train_loader,
        val_loader,
        EPOCHS_pre,
        optimizer,
        EARLY_STOPPING_PATIENCE_pre,
        lambda_val=LAMB,
        data_aug=DATA_AUG,
        tau=TAU,
        loss_recon=criterion_recon,
    )
    test_result = test_pre(
        model,
        test_loader,
        lambda_val=LAMB,
        data_aug=DATA_AUG,
        tau=TAU,
        loss_recon=criterion_recon,
    )

    create_plots_and_save_results(
        result,
        test_result,
        f"{run}_{pre_model_name}",
        best_model,
        PRE_RESULTS_PATH,
        PRE_MODEL_PATH,
    )

    results.append(result)
    test_results.append(test_result)

save_average_results(results, test_results, pre_model_name, PRE_RESULTS_PATH)
