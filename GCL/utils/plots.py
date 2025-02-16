import pandas as pd
import matplotlib.pyplot as plt
import os
import torch


def solve_paths_pre(SAVED_MODELS_PATH_pre, RESULTS_PATH_pre, DATA_AUG, LAMBDA_VAL):

    PRE_MODEL_PATH = f"{SAVED_MODELS_PATH_pre}/data_aug_{DATA_AUG}_{LAMBDA_VAL}"
    PRE_RESULTS_PATH = f"{RESULTS_PATH_pre}/data_aug_{DATA_AUG}_{LAMBDA_VAL}"
    os.makedirs(f"{PRE_MODEL_PATH}", exist_ok=True)
    os.makedirs(f"{PRE_RESULTS_PATH}", exist_ok=True)
    return PRE_MODEL_PATH, PRE_RESULTS_PATH


def solve_paths_ft(SAVED_MODELS_PATH_ft, RESULTS_PATH_ft, DATA_AUG, LAMBDA_VAL):

    FT_MODEL_PATH = f"{SAVED_MODELS_PATH_ft}/data_aug_{DATA_AUG}_{LAMBDA_VAL}"
    FT_RESULTS_PATH = f"{RESULTS_PATH_ft}/data_aug_{DATA_AUG}_{LAMBDA_VAL}"
    os.makedirs(f"{FT_MODEL_PATH}", exist_ok=True)
    os.makedirs(f"{FT_RESULTS_PATH}", exist_ok=True)
    return FT_MODEL_PATH, FT_RESULTS_PATH


def save_average_results(results, test_results, filename, path):
    # Para results:
    # Inicializa un diccionario para guardar los promedios de la última época
    promedios_ultima_epoca = {}

    # Itera a través de las claves (métricas) en el primer elemento de la lista para establecer las métricas a calcular
    for clave in results[0].keys():
        # Calcula el promedio de la última época para cada métrica
        suma = sum(result[clave][-1] for result in results)
        promedio = suma / len(results)
        promedios_ultima_epoca[clave] = promedio

    df = pd.DataFrame([promedios_ultima_epoca])
    os.makedirs(f"{path}/promedios/train_and_val", exist_ok=True)
    df.to_excel(f"{path}/promedios/train_and_val/{filename}.xlsx", index=False)

    # Para test_results:
    averages = {
        key: sum(d[key] for d in test_results) / len(test_results)
        for key in test_results[0]
    }
    df = pd.DataFrame([averages])
    os.makedirs(f"{path}/promedios/test", exist_ok=True)
    df.to_excel(f"{path}/promedios/test/{filename}.xlsx", index=False)


def save_to_excel_train(data, filename, path):
    """
    Save data to an Excel file.

    Parameters:
    - data (dict): A dictionary where keys are column names and values are lists of numbers.
    - filename (str): The path and name of the Excel file to save the data to.
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame to an Excel file
    os.makedirs(f"{path}/train", exist_ok=True)
    df.to_excel(f"{path}/train/{filename}.xlsx", index=False)
    return filename


def save_to_excel_test(data, filename, path):
    """
    Save data to an Excel file.

    Parameters:
    - data (dict): A dictionary where keys are column names and values are lists of numbers.
    - filename (str): The path and name of the Excel file to save the data to.
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data])
    # Save the DataFrame to an Excel file
    os.makedirs(f"{path}/test", exist_ok=True)
    df.to_excel(f"{path}/test/{filename}.xlsx", index=False)
    return filename


import matplotlib.pyplot as plt


def create_and_save_plots(data, filename, path):
    """
    Create a plot for each specified pair of metrics (train vs validation) if they exist in data
    and save them as a single image.

    Parameters:
    - data (dict): A dictionary where keys are metric names and values are lists of metric values per epoch.
    - filename (str): The path and name of the image file to save the plots to.
    """
    metrics_pairs = [
        ("train_loss", "val_loss"),
        ("train_accuracy", "val_accuracy"),
        ("train_loss_gender", "val_loss_gender"),
        ("train_loss_n_way", "val_loss_n_way"),
        ("train_f1", "val_f1"),
        ("train_mse", "val_mse"),  # Including MSE metrics conditionally
    ]

    # Filter out metric pairs that are not fully present in data
    valid_metrics_pairs = [
        (train, val) for train, val in metrics_pairs if train in data and val in data
    ]

    # Determine the number of rows needed based on the number of valid metric pairs
    n_rows = len(valid_metrics_pairs)
    n_cols = 1  # Columns are fixed at 1 since we're plotting train vs validation side by side

    # Create a figure to contain the plots with dynamic size
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 5))
    fig.suptitle("Training and Validation Metrics Comparison", fontsize=16)

    if n_rows > 1:
        axs_flat = axs.flat
    else:
        axs_flat = [axs] if n_rows == 1 else [fig.add_subplot(1, 1, 1)]

    # Plot each pair of valid metrics in its respective subplot
    for ax, (train_key, val_key) in zip(axs_flat, valid_metrics_pairs):
        ax.plot(data[train_key], label=f'Train {train_key.split("_")[1]}')
        ax.plot(data[val_key], label=f'Validation {val_key.split("_")[1]}')
        ax.set_title(f'{train_key.split("_")[1]} Comparison')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value")
        ax.legend()
        ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    os.makedirs(f"{path}/plots", exist_ok=True)
    plt.savefig(f"{path}/plots/{filename}.png")
    plt.close()  # Close the plot to save resources


def save_model(path, filename, best_model):
    model = best_model["model"]
    epoch = best_model["epoch"]
    optimizer = best_model["optimizer"]
    val_loss_avg = best_model["val_loss"]
    ckpt_name = f"{filename}.pt"
    full_path = f"{path}/{ckpt_name}"
    # save_model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss_avg,
        },
        full_path,
    )


def create_plots_and_save_results(
    results, results_test, filename, best_model, path_results, path_model
):
    save_to_excel_train(results, filename, path_results)
    create_and_save_plots(results, filename, path_results)
    save_to_excel_test(results_test, filename, path_results)
    save_model(path_model, f"{filename}_best_model", best_model)
