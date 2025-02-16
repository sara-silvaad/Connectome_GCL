# Connectome_GCL

This repository contains the code developed for the article Graph Contrastive Learning for Connectome Classification by Martín Schmidt, Sara Silva, Federico Larroca, Gonzalo Mateos, and Pablo Musé.


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

The file scs_desikan.mat contains the structural connectivity matrices, corr_matrices holds the functional connectivity matrices, and HCP_behavioral.csv includes demographic and behavioral information. All data are derived from the HCP(https://www.humanconnectome.org/study/hcp-young-adult).
      
# Running the code

1. Change DATA_PATH and FC_DOWNLOAD_NAME in the config files to match the directory structure where the data is stored. 

2. Run
   
   2.1 Graph Contrastive Learning (GCL)
   Navigate to the GCL folder and run the following command: python run_mains_supervised.py
   The script run_mains_supervised.py trains a Contrastive Learning model with and without data augmentation. It first performs the pre-training step, followed by the fine-tuning step.

   2.2 Graph Representation Learning (GRL) - without Contrastive Learning
   Navigate to the GRL folder and run one of the following commands:
   - python run_models_DA.py  # Train the model with data augmentation  
   - python run_models_GCN.py  # Train the model without data augmentation  


# Contact

Any questions regarding the code, please refer to Sara Silva at ssilva@fing.edu.uy or Martin Schmidt at mschmidt@fing.edu.uy
