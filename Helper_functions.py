import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_curve, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, auc
from contextlib import contextmanager
import gc
from torch_geometric.loader import NeighborLoader

try:
    from torch.amp import autocast as _autocast_new  # torch>=2.0 preferred API

    def _autocast_disabled(device_type: str):
        return _autocast_new(device_type=device_type, enabled=False)
except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as _autocast_old  # fallback for older torch

    def _autocast_disabled(device_type: str):
        return _autocast_old(enabled=False)

def calculate_scale_pos_weight(data):
    """
    Calculate the scale_pos_weight for imbalanced datasets.
    """
    #train_mask = data.train_mask
    y_train = data.y.cpu().numpy()
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    return float(neg) / float(pos)

def calculate_metrics(y_true, y_pred, y_pred_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_illicit = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0) # illicit is class 1
    recall = recall_score(y_true, y_pred, average='weighted')
    recall_illicit = recall_score(y_true, y_pred, pos_label=1, average='binary') # illicit is class 1
    f1 = f1_score(y_true, y_pred, average='weighted')
    f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary') # illicit is class 1
    
    roc_auc = roc_auc_score(y_true, y_pred_prob[:,1])  # assuming class 1 is the positive class
    roc_auc_illicit = roc_auc_score(y_true, y_pred_prob[:,1])
    
    PR_curve = precision_recall_curve(y_true, y_pred_prob[:,1])
    PRAUC = auc(PR_curve[1], PR_curve[0])
    
    kappa = cohen_kappa_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'precision_illicit': precision_illicit,
        'recall': recall,
        'recall_illicit': recall_illicit,
        'f1': f1,
        'f1_illicit': f1_illicit,
        'roc_auc': roc_auc,
        'roc_auc_illicit': roc_auc_illicit,
        'PR_curve': PR_curve,
        'PRAUC': PRAUC,
        'kappa': kappa,
    }
    
    return metrics

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            alpha_val = float(alpha)
            if not (0.0 <= alpha_val <= 1.0):
                raise ValueError("alpha float must lie in [0, 1]")
            self.alpha = torch.tensor([alpha_val, 1.0 - alpha_val], dtype=torch.float32)
        elif isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            raise TypeError("alpha must be None, float, sequence, or torch.Tensor")

        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.ndim != 1:
                raise ValueError("alpha tensor must be 1-dimensional")
            if torch.any(self.alpha < 0):
                raise ValueError("alpha tensor must be non-negative")
            if self.alpha.sum() == 0:
                raise ValueError("alpha tensor must have positive sum")
            self.alpha = self.alpha / self.alpha.sum()

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logits = inputs.float()
        targets = targets.long()
        device_type = 'cuda' if logits.is_cuda else 'cpu'
        with _autocast_disabled(device_type):
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prob of the true class
        
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        # More stable computation of pt
        pt = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0-1e-7)  # Prevent underflow/overflow
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha correctly depending on type
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]  # per-class weights
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
    
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Note: avoid top-level imports from `models` to prevent circular imports with `models.py`.
# Import model classes and training helpers locally inside functions where they're needed.
def balanced_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights (sum to 1) for 1-D integer labels.

    Unlabelled entries (label < 0) are ignored.
    """
    if labels.ndim != 1:
        labels = labels.view(-1)
    labels = labels.detach()
    valid = labels >= 0
    if not torch.any(valid):
        return torch.ones(num_classes, dtype=torch.float32) / float(num_classes)
    filtered = labels[valid].to(torch.long).cpu()
    counts = torch.bincount(filtered, minlength=num_classes).clamp_min(1)
    inv = (1.0 / counts.float())
    inv = inv / inv.sum()
    return inv

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

def _get_model_instance(trial, model, data, device):
    """
    Helper function to suggest hyperparameters and instantiate a model
    based on the model name.
    """
    if model == 'MLP':
        from models import MLP
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        return MLP(num_node_features=data.num_node_features, num_classes=2, hidden_units=hidden_units)
    
    elif model == 'SVM':
        from sklearn.svm import SVC
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        degree = trial.suggest_int('degree', 2, 5)
        C = trial.suggest_float('C', 0.1, 10.0, log=True)
        return SVC(kernel=kernel, C=C, class_weight='balanced', degree=degree, probability=True)

    elif model == 'XGB':
        max_depth = trial.suggest_int('max_depth', 5, 15)
        Gamma_XGB = trial.suggest_float('Gamma_XGB', 0, 5)
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        learning_rate_XGB = trial.suggest_float('learning_rate_XGB', 0.005, 0.05, log=False) # XGB learning rate
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        return XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=calculate_scale_pos_weight(data),
            learning_rate=learning_rate_XGB,
            max_depth=max_depth,
            n_estimators=n_estimators,
            colsample_bytree=colsample_bytree,
            gamma=Gamma_XGB,
            subsample=subsample,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        )

    elif model == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 15)
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced')

    elif model == 'GCN':
        from models import GCN
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        return GCN(num_node_features=data.x.shape[1], num_classes=2, hidden_units=hidden_units)

    elif model == 'GAT':
        from models import GAT
        hidden_units = trial.suggest_int('hidden_units', 32, 150)
        num_heads = trial.suggest_int('num_heads', 1, 8)
        return GAT(num_node_features=data.x.shape[1], num_classes=2, hidden_units=hidden_units, num_heads=num_heads)

    elif model == 'GIN':
        from models import GIN
        hidden_units = trial.suggest_int('hidden_units', 32, 128)
        return GIN(num_node_features=data.x.shape[1], num_classes=2, hidden_units=hidden_units)

    else:
        raise ValueError(f"Unknown model: {model}")

def _run_wrapper_model_test(model_name, data, params, criterion, early_stop_args, device, train_perf_eval, val_perf_eval, test_perf_eval, batch_size=4096):
    """
    Helper to run the final test for MLP, GCN, GAT, and GIN models using NeighborLoaders.
    """
    hidden_units = params.get("hidden_units", 64)
    learning_rate = params.get("learning_rate", 0.005)
    weight_decay = params.get("weight_decay", 0.0001)

    # Import model classes locally
    from models import MLP, GCN, GAT, GIN

    if model_name == "MLP":
        model = MLP(num_node_features=data.num_features, num_classes=2, hidden_units=hidden_units).to(device)
    elif model_name == "GCN":
        model = GCN(num_node_features=data.num_features, num_classes=2, hidden_units=hidden_units).to(device)
    elif model_name == "GAT":
        num_heads = params.get("num_heads", 4)
        model = GAT(num_node_features=data.num_features, num_classes=2, hidden_units=hidden_units, num_heads=num_heads).to(device)
    elif model_name == "GIN":
        model = GIN(num_node_features=data.num_features, num_classes=2, hidden_units=hidden_units).to(device)
    else:
        raise ValueError(f"Invalid wrapper model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    from models import ModelWrapper
    model_wrapper = ModelWrapper(model=model, optimizer=optimizer, criterion=criterion)

    # --- NEW: Create Loaders for the final test run ---
    # We use the same neighbor sampling parameters as in the optimization loop
    train_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=batch_size, input_nodes=train_perf_eval, shuffle=True, num_workers=4)
    val_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=batch_size, input_nodes=val_perf_eval, shuffle=False, num_workers=4)
    test_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=batch_size, input_nodes=test_perf_eval, shuffle=False, num_workers=4)

    from training_and_testing import train_and_test

    # Pass the loaders instead of the raw data object
    return train_and_test(
        model_wrapper=model_wrapper,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        **early_stop_args
    )
    
def train_and_test_NMW_models(model_name, data, train_perf_eval, val_perf_eval, test_perf_eval, params_for_model):
    match model_name:
        case "SVM":
            C = params_for_model.get("C", 1.0)
            degree = params_for_model.get("degree", 3)
            kernel = params_for_model.get("kernel", 'rbf')
            svm_model = SVC(kernel=kernel, C=C, class_weight='balanced', degree=degree, probability=True)
            combined_mask = train_perf_eval | val_perf_eval
            x_train = data.x[combined_mask].detach().cpu().numpy()
            y_train = data.y[combined_mask].detach().cpu().numpy()
            x_test = data.x[test_perf_eval].detach().cpu().numpy()
            y_test = data.y[test_perf_eval].detach().cpu().numpy()
            #scaler = StandardScaler()
            #x_train = scaler.fit_transform(x_train)
            #x_test = scaler.transform(x_test)
            svm_model.fit(x_train, y_train)
            del x_train, y_train
            gc.collect()
            pred = svm_model.predict(x_test)
            prob = svm_model.predict_proba(x_test)
            del x_test
            metrics = calculate_metrics(y_test, pred, prob)
            return metrics
        case "RF":
            n_estimators = params_for_model.get("n_estimators", 100)
            max_depth = params_for_model.get("max_depth", 10)
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced')
            combined_mask = train_perf_eval | val_perf_eval
            x_train = data.x[combined_mask].detach().cpu().numpy()
            y_train = data.y[combined_mask].detach().cpu().numpy()
            x_test = data.x[test_perf_eval].detach().cpu().numpy()
            y_test = data.y[test_perf_eval].detach().cpu().numpy()
            #scaler = StandardScaler()
            #x_train = scaler.fit_transform(x_train)
            #x_test = scaler.transform(x_test)
            rf_model.fit(x_train, y_train)
            del x_train, y_train
            gc.collect()
            pred = rf_model.predict(x_test)
            prob = rf_model.predict_proba(x_test)
            del x_test
            metrics = calculate_metrics(y_test, pred, prob)
            del rf_model
            gc.collect()
            return metrics
        case "XGB":
            from xgboost import XGBClassifier
            max_depth = params_for_model.get("max_depth", 10)
            n_estimators = params_for_model.get("n_estimators", 100)
            combined_mask = train_perf_eval | val_perf_eval
            x_train = data.x[combined_mask].detach().cpu().numpy()
            y_train = data.y[combined_mask].detach().cpu().numpy()
            x_test = data.x[test_perf_eval].detach().cpu().numpy()
            y_test = data.y[test_perf_eval].detach().cpu().numpy()
            #scaler = StandardScaler()
            #x_train = scaler.fit_transform(x_train)
            #x_test = scaler.transform(x_test)
            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            scale_pos_weight = float(neg) / max(1.0, float(pos))
            xgb_model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, scale_pos_weight=scale_pos_weight)
            xgb_model.fit(x_train, y_train)
            del x_train, y_train
            gc.collect()
            pred = xgb_model.predict(x_test)
            prob = xgb_model.predict_proba(x_test)
            del x_test
            metrics = calculate_metrics(y_test, pred, prob)
            del xgb_model
            gc.collect()
            return metrics
        
@contextmanager
def inference_mode_if_needed(model_name: str):
    """
    Context manager that disables gradient tracking if the model is CPU-based
    or if we are in evaluation mode.
    """
    if model_name in ["SVM", "XGB", "RF"]:
        with torch.no_grad():
            yield
    else:
        yield

def run_trial_with_cleanup(trial_func, model_name, *args, **kwargs):
    """
    Runs a trial function safely with:
      - Automatic no_grad() for CPU-based models.
      - GPU/CPU memory cleanup after each trial.
    
    Parameters
    ----------
    trial_func : callable
        The trial function to run (e.g., objective).
    model_name : str
        Name of the model (MLP, SVM, XGB, RF, GCN, GAT, GIN).
    *args, **kwargs :
        Arguments to pass to trial_func.
        
    Returns
    -------
    result : Any
        The return value of the trial function.
    """
    try:
        with inference_mode_if_needed(model_name):
            result = trial_func(*args, **kwargs)
        return result
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        
def check_study_existence(model_name, data_for_optimization):
    """
    Check if an Optuna study exists and has a sufficient number of trials (>= 50).
    
    If a study exists but has fewer than 50 trials, it is automatically
    deleted.
    
    Parameters
    ----------
    model_name : str
        Name of the model (MLP, SVM, XGB, RF, GCN, GAT, GIN).
    data_for_optimization : str
        Name of the dataset used for optimization.
        
    Returns
    -------
    exists : bool
        True if a study exists with >= 50 trials, False otherwise.
    """
    import optuna
    study_name = f'{model_name}_optimization on {data_for_optimization} dataset'
    storage_url = 'sqlite:///optimization_results.db'
    
    try:
        # 1. Attempt to load the study
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        # 2. Study exists, check the number of trials (runs)
        num_trials = len(study.trials)
        
        if num_trials < 50:
            # 3. Less than 50 runs: Delete and restart.
            print(f"Study '{study_name}' found with {num_trials} trials (< 50). Deleting study to restart.")
            optuna.delete_study(study_name=study_name, storage=storage_url) 
            return False
        else:
            # 4. 50 or more runs: study is valid, return True
            print(f"Study '{study_name}' found with {num_trials} trials (>= 50). Study is valid.")
            return True
            
    except KeyError:
        # 5. Study does not exist: return False
        print(f"Study '{study_name}' not found.")
        return False
    
    
def print_gpu_tensors():
    print(f"{'Type':<20} | {'Size':<20} | {'Memory (MB)':<10}")
    print("-" * 60)
    total_mem = 0
    for obj in gc.get_objects():
        try:
            # Check for Tensors or Parameter objects
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    mem = obj.element_size() * obj.nelement() / 1024**2
                    total_mem += mem
                    print(f"{str(type(obj).__name__):<20} | {str(list(obj.size())):<20} | {mem:.2f}")
        except:
            pass
    print("-" * 60)
    print(f"Total Approximate Tensor Memory: {total_mem:.2f} MB")

def find_optimal_batch_size(model_builder, data, device, train_mask, num_neighbors=[10, 10]):
    """
    Finds the optimal batch size for NeighborLoader by testing increasing sizes
    until OOM, then binary searching.
    
    Args:
        model_builder: A function that returns a fresh model instance (on CPU).
        data: The data object.
        device: The device to train on.
        train_mask: Mask for training nodes.
        num_neighbors: Neighbor sampling configuration.
    
    Returns:
        int: Optimal batch size (approx 90% of max safe size).
    """
    print("Searching for optimal batch size...")
    
    low = 32768
    high = 1000000 # Start with a reasonable upper bound
    optimal = 65536 # Safe default
    
    # Define a simple training loop for testing
    def test_batch_size(batch_size):
        try:
            # Create a fresh model and move to device
            model = model_builder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            
            loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=train_mask,
                shuffle=True
            )
            
            model.train()
            # Run a few batches to ensure stability
            # We don't need to run a full epoch, just enough to fill memory buffers
            steps = 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                
                # Slice logic similar to ModelWrapper
                batch_size_actual = batch.batch_size
                out_sliced = out[:batch_size_actual]
                y_sliced = batch.y[:batch_size_actual]
                
                loss = criterion(out_sliced, y_sliced)
                loss.backward()
                optimizer.step()
                
                steps += 1
                if steps >= 5: # Test 5 batches
                    break
            
            del model, optimizer, criterion, loader, batch, out, out_sliced, loss
            if device == 'cuda':
                torch.cuda.empty_cache()
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size} OOM.")
                if device == 'cuda':
                     torch.cuda.empty_cache()
                return False
            else:
                raise e
        except Exception as e:
            print(f"Batch size {batch_size} failed with error: {e}")
            return False

    # 1. Exponential search to find upper bound
    current = low
    max_safe = low
    
    while current <= high:
        print(f"Testing batch size: {current}")
        if test_batch_size(current):
            max_safe = current
            current *= 2
        else:
            high = current
            break
            
    # 2. Binary search between max_safe and high (which failed)
    low = max_safe
    
    while low < high - 512: # Granularity of 512
        mid = (low + high) // 2
        print(f"Binary search testing: {mid}")
        if test_batch_size(mid):
            low = mid
        else:
            high = mid
            
    # Return 95% of the max found size to be safe
    optimal = int(low * 0.95)
    print(f"Optimal batch size found: {optimal}")
    return optimal