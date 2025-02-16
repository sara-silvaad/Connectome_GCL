import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from graspologic.plot import pairplot
from nilearn import plotting
import numpy as np
from matplotlib.pyplot import cm
import json
import multiprocessing as mp


human_labs = [
    "L",
    "R",
    "L",
    "R",
    "BS",
    "L",
    "R",
    "L",
    "R",
    "L",
    "R",
    "L",
    "R",
    "L",
    "R",
    "L",
    "R",
    "L",
    "R",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
]


coordinates = [
    [-9, 9, -8],  # empieza subcortical
    [9, 9, -8],
    [-22, -3, -18],
    [22, -3, -18],
    [0, -33, -23],  # brain stem
    [-13, 15, 9],
    [13, 15, 9],
    [-17, -56, -36],
    [17, -56, -36],
    [-12, -15, -8],
    [12, -15, -8],
    [-22, -25, -9],
    [22, -25, -9],
    [-21, 4, 0],
    [21, 4, 0],
    [-25, 8, 3],
    [25, 8, 3],
    [-12, -15, 8],
    [12, -15, 8],  # termina subcortical
    [-52.7, -44.5, 4.6],  # lBKS (ctx-lh-bankssts) - BA 22
    [-6.6, 18, 26.1],  # lCAC (ctx-lh-caudalanteriorcingulate) - BA 8
    [-34.6, 10.2, 42.8],  # lCMF (ctx-lh-caudalmiddlefrontal)
    [-8.7, -79.6, 18],  # lCUN (ctx-lh-cuneus)
    [-25.8, -7.6, -31.6],  # lENT (ctx-lh-entorhinal)
    [-35.7, -43.3, -19.7],  # lFUS (ctx-lh-fusiform)
    [-40, -66.4, 27.3],  # lIP (ctx-lh-inferiorparietal)
    [-48.9, -34.4, -22.2],  # lIT (ctx-lh-inferiortemporal)
    [-9.8, -44.8, 16.9],  # lIST (ctx-lh-isthmuscingulate)
    [-29.7, -86.9, -1],  # lLO (ctx-lh-lateraloccipital)
    [-24, 28.6, -14.4],  # lLOF (ctx-lh-lateralorbitofrontal)
    [-16.5, -66.8, -4.3],  # lLIN (ctx-lh-lingual)
    [-8, 34.9, -14.9],  # lMOF (ctx-lh-medialorbitofrontal)
    [-55.6, -31.1, -12.9],  # lMT (ctx-lh-middletemporal)
    [-24.7, -31.2, -17.4],  # lPHIP (ctx-lh-parahippocampal)
    [-10, -28.7, 56.1],  # lPARA (ctx-lh-paracentral)
    [-44.6, 14.6, 13.1],  # lPOP (ctx-lh-parsopercularis)
    [-41, 38.8, -11.1],  # lPOB (ctx-lh-parsorbitalis)
    [-42.4, 30.6, 2.3],  # lPT (ctx-lh-parstriangularis)
    [-13.9, -80.6, 6],  # lPERI (ctx-lh-pericalcarine)
    [-42.3, -23.8, 43.6],  # lPOC (ctx-lh-postcentral)
    [-7.3, -17.4, 35.7],  # lPCG (ctx-lh-posteriorcingulate)
    [-37.8, -10.7, 42.1],  # lPRC (ctx-lh-precentral)
    [-11.6, -57.5, 36.7],  # lPREC (ctx-lh-precuneus)
    [-6.8, 33.9, 1.6],  # lRAC (ctx-lh-rostralanteriorcingulate)
    [-31.3, 41.2, 16.5],  # lRMF (ctx-lh-rostralmiddlefrontal)
    [-12.6, 22.9, 42.4],  # lSF (ctx-lh-superiorfrontal)
    [-22.8, -60.9, 46.3],  # lSP (ctx-lh-superiorparietal)
    [-52.1, -17.8, -4.4],  # lST (ctx-lh-superiortemporal)
    [-50.4, -38.8, 31],  # lSUPRA (ctx-lh-supramarginal)
    [-8.6, 61.7, -8.7],  # lFP (ctx-lh-frontalpole)
    [-32.8, 8.4, -34.8],  # lTP (ctx-lh-temporalpole)
    [-44, -24.2, 6],  # lTRANS (ctx-lh-transversetemporal)
    [-34.2, -4.3, 2.2],  # lINS (ctx-lh-insula)
    [51.9, -40.6, 5.6],  # rBKS (ctx-rh-bankssts)
    [7.3, 18.7, 26.3],  # rCAC (ctx-rh-caudalanteriorcingulate)
    [34.9, 11.8, 43],  # rCMF (ctx-rh-caudalmiddlefrontal)
    [8.7, -80.1, 19],  # rCUN (ctx-rh-cuneus)
    [26.2, -6.8, -31.9],  # rENT (ctx-rh-entorhinal)
    [35.9, -43, -19.2],  # rFUS (ctx-rh-fusiform)
    [42.8, -60.9, 28.1],  # rIP (ctx-rh-inferiorparietal)
    [49.3, -31.7, -23],  # rIT (ctx-rh-inferiortemporal)
    [8.9, -45.4, 17.6],  # rIST (ctx-rh-isthmuscingulate)
    [30.3, -86.3, 0.5],  # rLO (ctx-rh-lateraloccipital)
    [23.6, 28.5, -15.2],  # rLOF (ctx-rh-lateralorbitofrontal)
    [16.8, -66.3, -3.6],  # rLIN (ctx-rh-lingual)
    [8.8, 35.7, -14.8],  # rMOF (ctx-rh-medialorbitofrontal)
    [55.9, -29.5, -12.9],  # rMT (ctx-rh-middletemporal)
    [26.1, -31.3, -16.2],  # rPHIP (ctx-rh-parahippocampal)
    [9.9, -27.4, 55.6],  # rPARAC (ctx-rh-paracentral)
    [44.9, 14.4, 14.2],  # rPOP (ctx-rh-parsopercularis)
    [42.1, 39.2, -10],  # rPOB (ctx-rh-parsorbitalis)
    [45, 29.7, 4.5],  # rPT (ctx-rh-parstriangularis)
    [14, -79.7, 6.7],  # rPERI (ctx-rh-pericalcarine)
    [41.6, -22.4, 43.8],  # rPOC (ctx-rh-postcentral)
    [7.6, -17.1, 36.2],  # rPCG (ctx-rh-posteriorcingulate)
    [36.8, -9.9, 43.5],  # rPRC (ctx-rh-precentral)
    [11.7, -56.5, 37.7],  # rPREC (ctx-rh-precuneus)
    [8, 33.5, 2.1],  # rRAC (ctx-rh-rostralanteriorcingulate)
    [32.3, 40.9, 17.3],  # rRMF (ctx-rh-rostralmiddlefrontal)
    [13.4, 24.7, 42],  # rSF (ctx-rh-superiorfrontal)
    [22.6, -59.5, 48.1],  # rSP (ctx-rh-superiorparietal)
    [53, -14, -5.5],  # rST (ctx-rh-superiortemporal)
    [50.6, -33.3, 30.7],  # rSUPRA (ctx-rh-supramarginal)
    [10.3, 61.1, -10],  # rFP (ctx-rh-frontalpole)
    [34, 8.4, -33.1],  # rTP (ctx-rh-temporalpole)
    [44.8, -22.4, 6.5],  # rTRANS (ctx-rh-transversetemporal)
    [35.1, -3.9, 2.4],  # rINS (ctx-rh-insula)
]

# [-52.7, -44.5, 4.6],  # lBKS (ctx-lh-bankssts) - BA 22 (Wernicke's area)
# [-6.6, 18, 26.1],     # lCAC (ctx-lh-caudalanteriorcingulate) - BA 8 (Frontal Eye Fields)
# [-34.6, 10.2, 42.8],  # lCMF (ctx-lh-caudalmiddlefrontal) - BA 9/46 (Dorsolateral Prefrontal Cortex)
# [-8.7, -79.6, 18],    # lCUN (ctx-lh-cuneus) - BA 17/18 (Primary Visual Cortex)
# [-25.8, -7.6, -31.6], # lENT (ctx-lh-entorhinal) - BA 28 (Entorhinal Cortex)
# [-35.7, -43.3, -19.7],# lFUS (ctx-lh-fusiform) - BA 37 (Fusiform Gyrus)
# [-40, -66.4, 27.3],   # lIP (ctx-lh-inferiorparietal) - BA 39/40 (Angular Gyrus)
# [-48.9, -34.4, -22.2],# lIT (ctx-lh-inferiortemporal) - BA 20/21 (Inferior Temporal Gyrus)
# [-9.8, -44.8, 16.9],  # lIST (ctx-lh-isthmuscingulate) - BA 23/31 (Posterior Cingulate Cortex)
# [-29.7, -86.9, -1],   # lLO (ctx-lh-lateraloccipital) - BA 18/19 (Secondary Visual Cortex)
# [-24, 28.6, -14.4],   # lLOF (ctx-lh-lateralorbitofrontal) - BA 11 (Orbitofrontal Cortex)
# [-16.5, -66.8, -4.3], # lLIN (ctx-lh-lingual) - BA 19 (Visual Association Cortex)
# [-8, 34.9, -14.9],    # lMOF (ctx-lh-medialorbitofrontal) - BA 10/11 (Medial Prefrontal Cortex)
# [-55.6, -31.1, -12.9],# lMT (ctx-lh-middletemporal) - BA 21 (Middle Temporal Gyrus)
# [-24.7, -31.2, -17.4],# lPHIP (ctx-lh-parahippocampal) - BA 36/37 (Parahippocampal Gyrus)
# [-10, -28.7, 56.1],   # lPARA (ctx-lh-paracentral) - BA 4/6 (Primary Motor Cortex)
# [-44.6, 14.6, 13.1],  # lPOP (ctx-lh-parsopercularis) - BA 44/45 (Broca's Area)
# [-41, 38.8, -11.1],   # lPOB (ctx-lh-parsorbitalis) - BA 47 (Orbitofrontal Cortex)
# [-42.4, 30.6, 2.3],   # lPT (ctx-lh-parstriangularis) - BA 45 (Broca's Area)
# [-13.9, -80.6, 6],    # lPERI (ctx-lh-pericalcarine) - BA 17 (Primary Visual Cortex)
# [-42.3, -23.8, 43.6], # lPOC (ctx-lh-postcentral) - BA 3/1/2 (Primary Somatosensory Cortex)
# [-7.3, -17.4, 35.7],  # lPCG (ctx-lh-posteriorcingulate) - BA 23/31 (Posterior Cingulate Cortex)
# [-37.8, -10.7, 42.1], # lPRC (ctx-lh-precentral) - BA 6 (Premotor Cortex)
# [-11.6, -57.5, 36.7], # lPREC (ctx-lh-precuneus) - BA 7 (Precuneus)
# [-6.8, 33.9, 1.6],    # lRAC (ctx-lh-rostralanteriorcingulate) - BA 24/32 (Anterior Cingulate Cortex)
# [-31.3, 41.2, 16.5],  # lRMF (ctx-lh-rostralmiddlefrontal) - BA 10 (Anterior Prefrontal Cortex)
# [-12.6, 22.9, 42.4],  # lSF (ctx-lh-superiorfrontal) - BA 9 (Superior Frontal Gyrus)
# [-22.8, -60.9, 46.3], # lSP (ctx-lh-superiorparietal) - BA 7 (Superior Parietal Lobule)
# [-52.1, -17.8, -4.4], # lST (ctx-lh-superiortemporal) - BA 22 (Superior Temporal Gyrus)
# [-50.4, -38.8, 31],   # lSUPRA (ctx-lh-supramarginal) - BA 40 (Supramarginal Gyrus)
# [-8.6, 61.7, -8.7],   # lFP (ctx-lh-frontalpole) - BA 10 (Frontal Pole)
# [-32.8, 8.4, -34.8],  # lTP (ctx-lh-temporalpole) - BA 38 (Temporal Pole)
# [-44, -24.2, 6],      # lTRANS (ctx-lh-transversetemporal) - BA 41/42 (Primary Auditory Cortex)
# [-34.2, -4.3, 2.2],   # lINS (ctx-lh-insula) - BA 13/14 (Insular Cortex)
# [51.9, -40.6, 5.6],   # rBKS (ctx-rh-bankssts) - BA 22 (Wernicke's area)
# [7.3, 18.7, 26.3],    # rCAC (ctx-rh-caudalanteriorcingulate) - BA 8 (Frontal Eye Fields)
# [34.9, 11.8, 43],     # rCMF (ctx-rh-caudalmiddlefrontal) - BA 9/46 (Dorsolateral Prefrontal Cortex)
# [8.7, -80.1, 19],     # rCUN (ctx-rh-cuneus) - BA 17/18 (Primary Visual Cortex)
# [26.2, -6.8, -31.9],  # rENT (ctx-rh-entorhinal) - BA 28 (Entorhinal Cortex)
# [35.9, -43, -19.2],   # rFUS (ctx-rh-fusiform) - BA 37 (Fusiform Gyrus)
# [42.8, -60.9, 28.1],  # rIP (ctx-rh-inferiorparietal) - BA 39/40 (Angular Gyrus)
# [49.3, -31.7, -23],   # rIT (ctx-rh-inferiortemporal) - BA 20/21 (Inferior Temporal Gyrus)
# [8.9, -45.4, 17.6],   # rIST (ctx-rh-isthmuscingulate) - BA 23/31 (Posterior Cingulate Cortex)
# [30.3, -86.3, 0.5],   # rLO (ctx-rh-lateraloccipital) - BA 18/19 (Secondary Visual Cortex)
# [23.6, 28.5, -15.2],  # rLOF (ctx-rh-lateralorbitofrontal) - BA 11 (Orbitofrontal Cortex)
# [16.8, -66.3, -3.6],  # rLIN (ctx-rh-lingual) - BA 19 (Visual Association Cortex)
# [8.8, 35.7, -14.8],   # rMOF (ctx-rh-medialorbitofrontal) - BA 10/11 (Medial Prefrontal Cortex)
# [55.9, -29.5, -12.9], # rMT (ctx-rh-middletemporal) - BA 21 (Middle Temporal Gyrus)
# [26.1, -31.3, -16.2], # rPHIP (ctx-rh-parahippocampal) - BA 36/37 (Parahippocampal Gyrus)
# [9.9, -27.4, 55.6],   # rPARAC (ctx-rh-paracentral) - BA 4/6 (Primary Motor Cortex)
# [44.9, 14.4, 14.2],   # rPOP (ctx-rh-parsopercularis) - BA 44/45 (Broca's Area)
# [42.1, 39.2, -10],    # rPOB (ctx-rh-parsorbitalis) - BA 47 (Orbitofrontal Cortex)
# [45, 29.7, 4.5],      # rPT (ctx-rh-parstriangularis) - BA 45 (Broca's Area)
# [14, -79.7, 6.7],     # rPERI (ctx-rh-pericalcarine) - BA 17 (Primary Visual Cortex)
# [41.6, -22.4, 43.8],  # rPOC (ctx-rh-postcentral) - BA 3/1/2 (Primary Somatosensory Cortex)
# [7.6, -17.1, 36.2],   # rPCG (ctx-rh-posteriorcingulate) - BA 23/31 (Posterior Cingulate Cortex)
# [36.8, -9.9, 43.5],   # rPRC (ctx-rh-precentral) - BA 6 (Premotor Cortex)
# [11.7, -56.5, 37.7],  # rPREC (ctx-rh-precuneus) - BA 7 (Precuneus)
# [8, 33.5, 2.1],       # rRAC (ctx-rh-rostralanteriorcingulate) - BA 24/32 (Anterior Cingulate Cortex)
# [32.3, 40.9, 17.3],   # rRMF (ctx-rh-rostralmiddlefrontal) - BA 10 (Anterior Prefrontal Cortex)
# [13.4, 24.7, 42],     # rSF (ctx-rh-superiorfrontal) - BA 9 (Superior Frontal Gyrus)
# [22.6, -59.5, 48.1],  # rSP (ctx-rh-superiorparietal) - BA 7 (Superior Parietal Lobule)
# [53, -14, -5.5],      # rST (ctx-rh-superiortemporal) - BA 22 (Superior Temporal Gyrus)
# [50.6, -33.3, 30.7],  # rSUPRA (ctx-rh-supramarginal) - BA 40 (Supramarginal Gyrus)
# [10.3, 61.1, -10],    # rFP (ctx-rh-frontalpole) - BA 10 (Frontal Pole)
# [34, 8.4, -33.1],     # rTP (ctx-rh-temporalpole) - BA 38 (Temporal Pole)
# [44.8, -22.4, 6.5],   # rTRANS (ctx-rh-transversetemporal) - BA 41/42 (Primary Auditory Cortex)
# [35.1, -3.9, 2.4]     # rINS (ctx-rh-insula) - BA 13/14 (Insular Cortex)


node_coords = np.array(coordinates)


def save_average_results(results, test_results, filename, path):
    # Inicializa un diccionario para guardar los promedios de la última época
    promedios_ultima_epoca = {}

    # Itera a través de las claves (métricas) en el primer elemento de la lista para establecer las métricas a calcular
    for clave in results[0].keys():
        # Calcula el promedio de la última época para cada métrica
        suma = sum(result[clave][-1] for result in results)
        promedio = suma / len(results)
        promedios_ultima_epoca[clave] = promedio

    df = pd.DataFrame([promedios_ultima_epoca])
    os.makedirs(f"{path}/promedios", exist_ok=True)
    df.to_excel(f"{path}/promedios/{filename}.xlsx", index=False)

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


# Reemplaza 'data' con tu diccionario de datos actual y 'filename' con la ruta y nombre de archivo deseado
# create_and_save_plots(data, 'metrics_comparison.png')


def create_plots_and_save_results(results, results_test, filename, path):
    save_to_excel_train(results, filename, path)
    create_and_save_plots(results, filename, path)
    save_to_excel_test(results_test, filename, path)
    # save_model(path, filename, best_model)


def save_model(path, filename, best_model):
    model = best_model["model"]
    epoch = best_model["epoch"]
    optimizer = best_model["optimizer"]
    val_accuracy_avg = best_model["val_accuracy"]
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
            "accuracy": val_accuracy_avg,
        },
        full_path,
    )


def generate_embedding_figures(x, writer, epoch, train_or_val):

    B = x.shape[0] // 87
    D_pe = x.shape[1]
    x = x.view(B, 87, D_pe)
    # idx = random.randint(0, B)
    un_sujeto = x[0, :, :].cpu().detach().numpy()

    pairgrid = pairplot(un_sujeto, title=None, labels=human_labs)
    figure = pairgrid.figure
    writer.add_figure(
        f"Pairplot de señal en {train_or_val}",
        figure,
        global_step=epoch,
        close=True,
        walltime=None,
    )

    for embedding_dim in range(D_pe):

        node_vmin = np.min(un_sujeto[:, embedding_dim]) - 0.01
        node_vmax = np.max(un_sujeto[:, embedding_dim]) + 0.01
        display = plotting.plot_markers(
            un_sujeto[:, embedding_dim],
            node_coords,
            node_size="auto",
            node_cmap=cm.YlOrRd,
            node_vmin=node_vmin,
            node_vmax=node_vmax,
        )

        fig = display.frame_axes.figure

        writer.add_figure(
            f"Dimensión {embedding_dim} de señal en {train_or_val}",
            fig,
            global_step=epoch,
            close=True,
            walltime=None,
        )

    return writer


def save_inference_results(test_results, filename, save_path):
    file_path = os.path.join(save_path, f"{filename}.json")

    # Guardar los resultados en formato JSON
    with open(file_path, "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"Resultados de inferencia guardados en {file_path}")
    return


node_coords_87 = np.array(coordinates)


def generate_embedding_figures_87(x):

    p = np.maximum(abs(np.max(x)), abs(np.min(x)))
    node_vmin = -p
    node_vmax = p
    _ = plotting.plot_markers(
        x,
        node_coords_87,
        node_size="auto",
        node_cmap=mpl.colormaps["bwr"],
        node_vmin=node_vmin,
        node_vmax=node_vmax,
    )


node_coords_68 = np.array(coordinates[19:])


def generate_embedding_figures_68(x):

    p = np.maximum(abs(np.max(x)), abs(np.min(x)))
    node_vmin = -p
    node_vmax = p
    _ = plotting.plot_markers(
        x,
        node_coords_68,
        node_size="auto",
        node_cmap=mpl.colormaps["bwr"],
        node_vmin=node_vmin,
        node_vmax=node_vmax,
    )
