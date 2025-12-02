import torch
import gc
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from Helper_functions import calculate_metrics
from models import ModelWrapper, MLP



# In training_and_testing.py

def train_and_validate(
    model_wrapper,
    train_loader,  # Updated: expects loader, not full data
    val_loader,    # Updated: new argument for validation loader
    num_epochs,
    best_f1=-1,
    best_f1_model_wts=None,
    patience=None,
    min_delta=0.0,
    log_early_stop=False,
    **kwargs # Safely ignores old arguments like 'train_mask' or 'data' if passed by accident
):
    # Device check removed because loaders keep data on CPU until the batch loop.
    
    metrics = {
        'accuracy': [],
        'precision_weighted': [],
        'precision_illicit': [],
        'recall': [],
        'recall_illicit': [],
        'f1': [],
        'f1_illicit': [],
        'roc_auc': [],
        'roc_auc_illicit': [],
        #'PR_curve': [],
        'PRAUC': [],
        'kappa': [] 
    }
    epochs_without_improvement = 0
    best_epoch = -1
    
    for epoch in range(num_epochs):
        # Updated: Pass the loader directly. 
        # The model_wrapper.train_step you updated earlier handles the looping.
        train_loss = model_wrapper.train_step(train_loader)

        # Updated: Pass the validation loader.
        # The model_wrapper.evaluate you updated earlier handles the looping.
        val_loss, val_metrics = model_wrapper.evaluate(val_loader)
        
        # --- The rest of the logic remains exactly the same ---
        metrics['accuracy'].append(val_metrics['accuracy'])
        metrics['precision_weighted'].append(val_metrics['precision'])
        metrics['precision_illicit'].append(val_metrics['precision_illicit'])
        metrics['recall'].append(val_metrics['recall'])
        metrics['recall_illicit'].append(val_metrics['recall_illicit'])
        metrics['f1'].append(val_metrics['f1'])
        metrics['f1_illicit'].append(val_metrics['f1_illicit'])
        metrics['roc_auc'].append(val_metrics['roc_auc'])
        metrics['roc_auc_illicit'].append(val_metrics['roc_auc_illicit'])
        #metrics['PR_curve'].append(val_metrics['PR_curve'])
        metrics['PRAUC'].append(val_metrics['PRAUC'])
        metrics['kappa'].append(val_metrics['kappa'])

        current_f1 = val_metrics['f1_illicit']
        improved = current_f1 > (best_f1 + min_delta)
        if improved:
            best_f1, best_f1_model_wts = update_best_weights(
                model_wrapper.model,
                best_f1,
                current_f1,
                best_f1_model_wts
            )
            epochs_without_improvement = 0
            best_epoch = epoch + 1  # keep 1-based for readability
        else:
            epochs_without_improvement += 1

        if patience and epochs_without_improvement >= patience:
            if log_early_stop:
                print(
                    f"Early stopping triggered at epoch {epoch + 1} "
                    f"(best F1: {best_f1:.4f} @ epoch {best_epoch})"
                )
            break

    return metrics, best_f1_model_wts, best_f1

import copy
def update_best_weights(model, best_f1, current_f1, best_f1_model_wts):
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_f1_model_wts = copy.deepcopy(model.state_dict())
    return best_f1, best_f1_model_wts

def train_and_test(
    model_wrapper,
    data,
    num_epochs=200,
    patience=None,
    min_delta=0.0,
    log_early_stop=False
):
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model_wrapper.model.to(device)
    #data = data.to(device)
    
    
    metrics, best_model_wts, best_f1 = train_and_validate(
        model_wrapper,
        data,
        num_epochs,
        patience=patience,
        min_delta=min_delta,
        log_early_stop=log_early_stop
    )
    
    model_wrapper.model.load_state_dict(best_model_wts)
    test_loss, test_metrics = model_wrapper.evaluate(data, data.test_perf_eval_mask)
    
    return test_metrics, best_f1

