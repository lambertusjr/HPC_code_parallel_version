# Final code for reproducibility paper
# 1 is fraudulent
#%% Settings for runs
seeded_run = False
prototyping = False
num_epochs = 200


#%% Setup
#Importing all packages
import platform
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
import gc
import torch_geometric.transforms as T


#torch.cuda.memory._record_memory_history(max_entries=100000)



# pc = platform.system()
# if pc == "Darwin":
#     os.chdir("/Users/lambertusvanzyl/Desktop/Reproducibility_paper")
# else:
#     os.chdir("/Users/Lambertus/Desktop/Reproducibility_paper")
    
# if seeded_run:
#     torch.manual_seed(42)
#     np.random.seed(42)
# else:
#     seed = np.random.SeedSequence().entropy

# Importing custom libraries
from pre_processing import EllipticDataset, IBMAMLDataset_HiSmall, IBMAMLDataset_LiSmall, IBMAMLDataset_HiMedium, IBMAMLDataset_LiMedium, AMLSimDataset
from models import GCN, ModelWrapper


#%% Reading in data and pre-processing
# if pc == "Darwin":
#     #Processing elliptic dataset
#     elliptic_data = EllipticDataset(root='/Users/lambertusvanzyl/Documents/Datasets/Elliptic_dataset')[0]
#     #Processing IBM AML dataset
#     IBM_data_HiSmall = IBMAMLDataset_HiSmall(root='/Users/lambertusvanzyl/Documents/Datasets/IBM_AML_dataset/HiSmall')[0]
#     IBM_data_LiSmall = IBMAMLDataset_LiSmall(root='/Users/lambertusvanzyl/Documents/Datasets/IBM_AML_dataset/LiSmall')[0]
#     IBM_data_HiMedium = IBMAMLDataset_HiMedium(root='/Users/lambertusvanzyl/Documents/Datasets/IBM_AML_dataset/HiMedium')[0]
#     IBM_data_LiMedium = IBMAMLDataset_LiMedium(root='/Users/lambertusvanzyl/Documents/Datasets/IBM_AML_dataset/LiMedium')[0]
#     #Processing AMLSim dataset
#     AMLSim_data = AMLSimDataset(root='/Users/lambertusvanzyl/Documents/Datasets/AMLSim_dataset')[0]
# else:
#     #Processing elliptic dataset
#     #elliptic_data = EllipticDataset(root='/Users/Lambertus/Desktop/Datasets/Elliptic_dataset')[0]
#     #Processing IBM AML dataset
#     IBM_data_HiSmall = IBMAMLDataset_HiSmall(root='/Users/Lambertus/Desktop/Datasets/IBM_AML_dataset/HiSmall')[0]
#     IBM_data_LiSmall = IBMAMLDataset_LiSmall(root='/Users/Lambertus/Desktop/Datasets/IBM_AML_dataset/LiSmall')[0]
#     #IBM_data_HiMedium = IBMAMLDataset_HiMedium(root='/Users/Lambertus/Desktop/Datasets/IBM_AML_dataset/HiMedium')[0]
#     #IBM_data_LiMedium = IBMAMLDataset_LiMedium(root='/Users/Lambertus/Desktop/Datasets/IBM_AML_dataset/LiMedium')[0]
#     #Processing AMLSim dataset
#     #AMLSim_data = AMLSimDataset(root='/Users/Lambertus/Desktop/Datasets/AMLSim_dataset')[0]

# %%
# from torch.optim import Adam
# if prototyping:
#     data = elliptic_data
#     #data = IBM_data
#     # testing whether pre processing worked
#     hidden_units = 64
#     learning_rate=0.05
#     loss = nn.CrossEntropyLoss()
#     model = GCN(num_node_features=data.x.shape[1], num_classes=2, hidden_units=hidden_units)
#     optimizer = Adam(model.parameters(), lr=learning_rate)
#     model_wrapper = ModelWrapper(model, optimizer, loss)
#     for i in range(1):
#         train_loss = model_wrapper.train_step(data, data.train_perf_eval_mask)
#         val_loss, val_metrics = model_wrapper.evaluate(data, data.val_perf_eval_mask)
#         print(f"Epoch {i+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1 illicit: {val_metrics['f1_illicit']:.4f}")

# %% Optuna runs
# %% Optuna runs
import sys
from hyperparameter_tuning import run_optimization

# Default to a specific dataset if no arg provided, or exit
dataset_name = sys.argv[1] if len(sys.argv) > 1 else "IBM_AML_HiMedium"

print(f"Running optimization for: {dataset_name}")

# Load ONLY the required data to save memory
match dataset_name:
    case "Elliptic":
        data_for_optimization = "Elliptic"
        data = EllipticDataset(root='/Users/lambertusvanzyl/Documents/Datasets/Elliptic_dataset')[0]
    case "IBM_AML_HiSmall":
        data_for_optimization = "IBM_AML_HiSmall"
        data = IBMAMLDataset_HiSmall(root='/Users/lambertusvanzyl/Documents/Datasets/IBM_AML_dataset/HiSmall')[0]
    case "IBM_AML_LiSmall":
        data_for_optimization = "IBM_AML_LiSmall"
        data = IBMAMLDataset_LiSmall(root='/Users/lambertusvanzyl/Documents/Datasets/IBM_AML_dataset/LiSmall')[0]
    case "IBM_AML_HiMedium":
        data_for_optimization = "IBM_AML_HiMedium"
        data = IBMAMLDataset_HiMedium(root='dataset/HiMedium')[0]
    case "IBM_AML_LiMedium":
        data_for_optimization = "IBM_AML_LiMedium"
        data = IBMAMLDataset_LiMedium(root='dataset/LiMedium')[0]
    case "AMLSim":
        data_for_optimization = "AMLSim"
        data = AMLSimDataset(root='/Users/lambertusvanzyl/Documents/Datasets/AMLSim_dataset')[0]

# Convert edge_index to sparsetensor to save memory
# import torch_geometric.transforms as T
# transform = T.ToSparseTensor(remove_edge_index=True)
# data = transform(data)

def save_testing_results_csv(results, path=f"{data_for_optimization}_testing_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(f"csv_results/{data_for_optimization}_testing_results.csv", index=False)

model_parameters, testing_results = run_optimization(
    models=['GCN', 'GAT', 'GIN', 'MLP'],
    data=data,
    train_perf_eval=data.train_perf_eval_mask,
    val_perf_eval=data.val_perf_eval_mask,
    test_perf_eval=data.test_perf_eval_mask,
    train_mask=data.train_mask,
    val_mask=data.val_mask,
    data_for_optimization=data_for_optimization
)

save_testing_results_csv(testing_results, path=f"{data_for_optimization}_testing_results.csv")
# %% Importing optuna trials from HPC runs to local PC
# import optuna
# # 1. Define your database paths (URLs)
# # Note: Use absolute paths if the files are in different folders
# pc_storage = "sqlite:///optimization_results.db"
# cluster_storage = "sqlite:////Users/lambertusvanzyl/Downloads/HPC_completed_run/optimization_results_HPC.db" # /Users/lambertusvanzyl/Downloads/HPC_completed_run/optimization_results_HPC.db

# hpc_studies = optuna.get_all_study_summaries(storage=cluster_storage)

# for study_summary in hpc_studies:
#     study_name = study_summary.study_name
#     print(f"Merging study: {study_name}...")
    
#     # 3. Copy each study from HPC file to Master file
#     # If the study name already exists in Master, trials are appended.
#     # If it doesn't exist, it is created.
#     optuna.copy_study(
#         from_study_name=study_name,
#         from_storage=cluster_storage,
#         to_storage=pc_storage,
#         to_study_name=study_name
#     )

# print("Migration complete. All HPC trials are now in your Master database.")




# %%
