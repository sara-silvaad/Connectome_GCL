from models.GCN_encoder_decoder_classifier import (
    EncoderDecoderSCFC,
    EncoderClassifierSC,
)

DATA_PATH = "/datos/projects/ssilva/data"

# data
ROOT_PATH = f"{DATA_PATH}/processed_datasets/"
MATRIX_FILE = f"{DATA_PATH}/original_data/scs_desikan.mat"
FC_PATHS = f"{DATA_PATH}/87_nodes_downloaded_on_imerl/corr_matrices/"

DATASET_NAME = "Gender"

LABELS_PATH = f"{DATA_PATH}/original_data/HCP_behavioral.csv"

MODEL = "EncoderClassifierSC"

model_mapping = {
    "EncoderDecoderSCFC": EncoderDecoderSCFC,
    "EncoderClassifierSC": EncoderClassifierSC,
}

# Only for EncoderDecoderSCFC
LAMB = 0.4

HIDDEN_DIMS = [87, 32, 16, 8]

TRAIN_PROPORTION = 1  # 0.1, 0.3, 0.5, 0.7

# training
EPOCHS = 10000
LR = 0.001
BATCH_SIZE = 64  # 8, 16, 32
EARLY_STOPPING_PATIENCE = 300
RUNS = 10

# results path
RESULTS_PATH = f"results_{TRAIN_PROPORTION}"
