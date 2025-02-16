# Connectome_GCL

This repository contains the code developed for the article Graph Contrastive Learning for Connectome Classification by Martín Schmidt, Sara Silva, Federico Larroca, Gonzalo Mateos, and Pablo Musé.

# Running the code

1. Change Data paths in config files -> DATA_PATH = "/datos/projects/ssilva/data", FC_DOWNLOAD_NAME

2. Run
   2.1 CL
   Stand on folder GCL and python run_mains_supervised.py with no arguments
   run_mains_supervised.py ...

   2.2 No CL
   Stand on folder GRL and python run_models_{}.py where {} can be one of DA or GCN to train the model with and without data augmentation respectively:
     run_models_DA.py
     run_models_GCN.py

# Data

"/datos/projects/ssilva/data" has the following folder structure:

| data
  | processed_datasets -> where datasets created by Dataset class will be stored
  | original_data 
    | scs_desikan.mat
    | HCP_behavioral.csv
  | {FC_DOWNLOAD_NAME}
    | corr_matrices
      |subject_0
      ...
      | subject_n

# Contact

Any questions regarding the code, please refer to Sara Silva at ssilva@fing.edu.uy or Martin Schmidt at mschmidt@fing.edu.uy
