import optuna
from optuna.trial import Trial
import numpy as np
from models import GCN, ModelWrapper
import torch
import torch.nn as nn
from Helper_functions import calculate_metrics, balanced_class_weights, FocalLoss, _get_model_instance, check_study_existence, run_trial_with_cleanup, train_and_test_NMW_models, _run_wrapper_model_test
from tqdm import tqdm, trange
from training_and_testing import train_and_validate, train_and_test
from torch.optim import Adam
import gc
from Helper_functions import print_gpu_tensors
from torch_geometric.loader import NeighborLoader
models = ['MLP', 'SVM', 'XGB', 'RF', 'GCN', 'GAT', 'GIN']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_EARLY_STOP = {
    "patience": 20,
    "min_delta": 1e-3,
}
EARLY_STOP_LOGGING = False


def _early_stop_args_from(source: dict) -> dict:
    """Build early stopping kwargs, falling back to defaults when keys are absent."""
    return {
        "patience": source.get("early_stop_patience", DEFAULT_EARLY_STOP["patience"]),
        "min_delta": source.get("early_stop_min_delta", DEFAULT_EARLY_STOP["min_delta"]),
        "log_early_stop": EARLY_STOP_LOGGING,
    }
    
def objective(trial, model, data, train_perf_eval, val_perf_eval, train_mask, val_mask):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    # Define models before try block to ensure they exist in 'locals()' for finally
    model_instance = None
    optimizer = None
    model_wrapper = None
    criterion = None

    try:
        # --- Shared Hyperparameters ---
        learning_rate = trial.suggest_float('learning_rate', 0.005, 0.05, log=False)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-3, log=True)
        
        # Note: data.y is on device, so .cpu() is needed for balanced_class_weights
        alpha_weights = balanced_class_weights(data.y[train_perf_eval].cpu()) 
        #loss_type = trial.suggest_categorical('loss_type', ['focal', 'cross_entropy'])
        loss_type = 'focal'
        if loss_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(weight=alpha_weights.to(device))
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01, log=False) #Previous learning rate range was too high for CE
        else:
            gamma_focal = trial.suggest_float('gamma_focal', 0.2, 5.0)
            alpha_weights = alpha_weights
            alpha_weights = alpha_weights
            criterion = FocalLoss(alpha=alpha_weights, gamma=gamma_focal)
            
        early_stop_patience = trial.suggest_int('early_stop_patience', 5, 40)
        early_stop_min_delta = trial.suggest_float('early_stop_min_delta', 1e-4, 5e-3, log=True)
        num_epochs = trial.suggest_int('num_epochs', 50, 250)
        trial_early_stop_args = _early_stop_args_from({
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta
        })

        # --- Model Instantiation (Refactored) ---
        model_instance = _get_model_instance(trial, model, data, device)
        train_loader = NeighborLoader(
                    data,
                    num_neighbors=[10, 10], # Sample 10 neighbors for each of the 2 GNN layers
                    batch_size=4096,        # Adjust this size to fill your GPU memory safely
                    input_nodes=train_mask
                )
        val_loader = NeighborLoader(
                    data,
                    num_neighbors=[10, 10], # Sample 10 neighbors for each of the 2 GNN layers
                    batch_size=4096,        # Adjust this size to fill your GPU memory safely
                    input_nodes=val_mask
                )
        # --- Training and Evaluation ---
        wrapper_models = ['MLP', 'GCN', 'GAT', 'GIN']
        sklearn_models = ['SVM', 'XGB', 'RF']
        
        
        # data = data.to(device) 
        # train_perf_eval = train_perf_eval.to(device) 
        # val_perf_eval = val_perf_eval.to(device) 
        #print_gpu_tensors()
        if model in wrapper_models:
            optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
            model_wrapper = ModelWrapper(model_instance, optimizer, criterion)
            model_wrapper.model.to(device)
            
            metrics, best_model_wts, best_f1 = train_and_validate(
                model_wrapper, train_loader, val_loader, num_epochs=num_epochs, **trial_early_stop_args
            )
            #print_gpu_tensors()
            torch.cuda.memory._dump_snapshot("Memory snapshot after training")
            return best_f1

        elif model in sklearn_models:
            # Data must be on CPU for sklearn
            train_x = data.x[train_perf_eval].cpu().numpy()
            train_y = data.y[train_perf_eval].cpu().numpy()
            val_x = data.x[val_perf_eval].cpu().numpy()
            val_y = data.y[val_perf_eval].cpu().numpy()

            model_instance.fit(train_x, train_y)
            pred = model_instance.predict(val_x)
            prob = model_instance.predict_proba(val_x)
            metrics = calculate_metrics(val_y, pred, prob)
            print_gpu_tensors()
            # Clean up large numpy arrays
            del train_x, train_y, val_x, val_y, pred, prob
            
            return metrics['f1_illicit']
    
    finally:
        print_gpu_tensors()
        # --- GUARANTEED CLEANUP ---
        # This block runs whether the trial succeeds, fails, or is pruned.
        if model_instance is not None:
            del model_instance
        if model_wrapper is not None:
            del model_wrapper
        if optimizer is not None:
            del optimizer
        if criterion is not None:
            del criterion
        if 'out' in locals():
            del out
        if 'pred' in locals():
            del pred
        if 'prob' in locals():
            del prob
        if 'best_model_wts' in locals():
            del best_model_wts
            
        if 'metrics' in locals():
            del metrics
        
        
        gc.collect() # Run Python's garbage collector
        if device == 'cuda':
            torch.cuda.empty_cache() # Clear PyTorch's CUDA cache
    
    # All other model types (XGBe+GIN, GINe+XGB) have been removed.

def run_trial_with_aggressive_cleanup(trial_func, *args, **kwargs):
    """
    More aggressive memory cleanup specifically for hyperparameter optimization
    """
    try:
        result = trial_func(*args, **kwargs)
        return result
    finally:
        # Force cleanup of any lingering tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache
        gc.collect(generation=2)  # Force full garbage collection
        
def run_optimization(models, data, train_perf_eval, val_perf_eval, test_perf_eval, train_mask, val_mask, data_for_optimization):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    
    # --- MOVE DATA AND MASKS TO DEVICE ONCE ---
    # data = data.to(device)
    # train_perf_eval = train_perf_eval.to(device)
    # val_perf_eval = val_perf_eval.to(device)
    # test_perf_eval = test_perf_eval.to(device)
    # train_mask and val_mask are not used in objective, move if needed
    # train_mask = train_mask.to(device) 
    # val_mask = val_mask.to(device)

    # --- Dynamically create result dictionaries ---
    model_parameters = {model_name: [] for model_name in models}
    
    METRIC_KEYS = [
        'accuracy', 'precision', 'precision_illicit', 'recall', 'recall_illicit',
        'f1', 'f1_illicit', 'roc_auc', 'roc_auc_illicit', 'PR_curve', 'PRAUC', 'kappa'
    ]
    testing_results = {
        model_name: {key: [] for key in METRIC_KEYS} 
        for model_name in models
    }
    
    # .cpu() is required as data.y is now on device
    focal_alpha = balanced_class_weights(data.y[train_perf_eval].cpu()) 
    focal_alpha_device = focal_alpha.to(device)

    for model_name in tqdm(models, desc="Models", unit="model"):
        sklearn_models = ['SVM', 'XGB', 'RF']
        if model_name in sklearn_models:
            n_trials = 70
        elif model_name == "MLP":
            n_trials = 100
        else:
            n_trials = 200
        study_name = f'{model_name}_optimization on {data_for_optimization} dataset'
        db_path = 'sqlite:///optimization_results.db'

        if check_study_existence(model_name, data_for_optimization): 
            study = optuna.load_study(study_name=study_name, storage=db_path)
        else:
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=db_path,
                load_if_exists=True
            )
            with tqdm(total=n_trials, desc=f"{model_name} trials", leave=False, unit="trial") as trial_bar:
                def _optuna_progress_callback(study, trial):
                    trial_bar.update()
                
                # Note: data, train_perf_eval, etc., are now the device tensors
                study.optimize(
                    lambda trial: run_trial_with_aggressive_cleanup( 
                        objective, trial, model_name, data, train_perf_eval, val_perf_eval, train_mask, val_mask
                    ),
                    n_trials=n_trials,
                    callbacks=[_optuna_progress_callback]
                )

        print(f"Best hyperparameters for {model_name}:", study.best_params)
        model_parameters[model_name].append(study.best_params)
        params_for_model = study.best_params

        # --- Setup for final test runs ---
        loss_type = params_for_model.get("loss_type", "focal")
        if loss_type == "cross_entropy":
            criterion = nn.CrossEntropyLoss(weight=focal_alpha_device)
        else:
            gamma_focal = params_for_model.get("gamma_focal", 2.0)
            # Use focal_alpha_device, not focal_alpha
            criterion = FocalLoss(alpha=focal_alpha_device, gamma=gamma_focal) 
        early_stop_args = _early_stop_args_from(params_for_model)

        # --- Final Test Runs (Refactored) ---
        for _ in trange(30, desc=f"Runs for {model_name}", leave=False, unit="run"):
            test_metrics = {}
            best_f1 = None # Not all test runs return this

            match model_name:
                case "MLP" | "GCN" | "GAT" | "GIN":
                    test_metrics, best_f1 = _run_wrapper_model_test(
                        model_name, data, params_for_model, criterion, early_stop_args, device,
                        train_perf_eval, val_perf_eval, test_perf_eval
                    )
                case "SVM" | "RF" | "XGB":
                    test_metrics = train_and_test_NMW_models(
                        model_name=model_name,
                        data=data, # This function will need to do .cpu() internally
                        train_perf_eval=train_perf_eval,
                        val_perf_eval=val_perf_eval,
                        test_perf_eval=test_perf_eval,
                        params_for_model=params_for_model
                    )
                case _:
                    print(f"Warning: No test logic defined for {model_name}. Skipping.")
                    continue
            print(f"Test metrics for {model_name} run:", test_metrics)
            # --- Centralized Metric Appending ---
            for key in METRIC_KEYS:
                if key in test_metrics:
                    testing_results[model_name][key].append(test_metrics[key])

            # --- CLEANUP *INSIDE* TEST LOOP ---
            # Assumes _run_wrapper_model_test & train_and_test_NMW_models
            # create and destroy their own models. This is extra insurance.
            del test_metrics
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

        # --- CLEANUP *AFTER* 30-RUN LOOP ---
        # Clean up before starting the next model's optimization
        del params_for_model
        del criterion
        del early_stop_args
        del study # Free study object, it's saved to DB
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    # --- FINAL CLEANUP ---
    # Move data back to CPU if needed elsewhere, or delete
    del data, train_perf_eval, val_perf_eval, test_perf_eval
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return model_parameters, testing_results