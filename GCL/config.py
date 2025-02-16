from models.GCN_encoder_decoder_classifier import (
    GCL_FeatureMaskingEdgeDropping,
    GCL_FeatureMaskingEdgeDroppingDecoder,
    GCL_NoDA,
    GCL_NoDA_Decoder,
)
import torch

DATA_PATH = "/datos/projects/ssilva/data"

# data
ROOT_PATH = f"{DATA_PATH}/processed_datasets"
MATRIX_FILE = f"{DATA_PATH}/original_data/scs_desikan.mat"
FC_PATHS = f"{DATA_PATH}/87_nodes_downloaded_on_imerl/corr_matrices/"

DATASET_NAME = "Gender"

LABELS_PATH = f"{DATA_PATH}/original_data/HCP_behavioral.csv"


# model
DATA_AUG = "featureMasking_edgeDropping"

model_mapping = {
    "featureMasking_edgeDropping": (GCL_FeatureMaskingEdgeDropping, None),
    "featureMasking_edgeDropping_Decoder": (
        GCL_FeatureMaskingEdgeDroppingDecoder,
        torch.nn.MSELoss(),
    ),
    "NoDA": (GCL_NoDA, None),
    "NoDA_Decoder": (GCL_NoDA_Decoder, torch.nn.MSELoss()),
}

HIDDEN_DIMS = [87, 32, 16, 8]
TAU = 1
LAMB = 0.25

TRAIN_PROPORTION = 1  # 0.1, 0.3, 0.5, 0.7

# training
# pre training
EPOCHS_pre = 5 # 10000
LR_pre = 0.001
BATCH_SIZE_pre = 128  # 16, 32, 64, 128
EARLY_STOPPING_PATIENCE_pre = 200
RUNS_pre = 1
SAVE_MODEL_pre = True

# fine tune
EPOCHS_ft = 5 # 200
EPOCHS_all_ft = 100
LR_ft = 0.001
BATCH_SIZE_ft = 16  # 4, 8, 16, 16
EARLY_STOPPING_PATIENCE_ft = 100
RUNS_ft = 5
SAVE_MODEL_ft = True
SELECTED_RUN = 0

NORMALIZE_EMB = True
FT_HIDDEN_DIM = 1

# results path pretrained
SAVED_MODELS_PATH_pre = f"saved_models/PretrainedGCL_{TRAIN_PROPORTION}"
RESULTS_PATH_pre = f"results/PretrainedGCL_{TRAIN_PROPORTION}"

# results path ft
SAVED_MODELS_PATH_ft = f"saved_models/FinetuneGCL_{TRAIN_PROPORTION}"
RESULTS_PATH_ft = f"results/FinetuneGCL_{TRAIN_PROPORTION}"
